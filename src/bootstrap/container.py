"""The container: builds every subsystem and attaches it to `app.state`."""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime

from fastapi import FastAPI

from core.config import load_full_config
from core.logging_config import logger
from domain.policies.room_policy import room_line
from domain.policies.token_budget_policy import PromptBudgets, estimate_tokens
from service.agent.assistant_orchestrator import AssistantOrchestrator
from service.agent.control_decoder import ControlDecoder
from service.agent.registry import AgentRegistry
from service.agent.responder import ResponderAgent
from service.agent.supervisor import SupervisorAgent
from service.agent.tool_use_guard import ToolUseGuard
from service.approval.approval_broker import ApprovalBroker
from service.conversation.conversation_manager import ConversationManager
from service.conversation.summariser import ConversationSummariser
from service.governance.factory import build_governance
from service.memory.knowledge_base import KnowledgeBase
from service.privacy.capture_gate import CaptureGate
from service.prompting.prompt_composer import PromptComposer
from service.sensing.ambient_dispatcher import AmbientDispatcher
from service.sensing.event_bus import EventBus
from service.session.session_state import SessionStateService
from tpa.filestore.file_prompt_store import FilePromptStore
from tpa.filestore.json_knowledge_store import JsonKnowledgeStore
from tpa.filestore.yaml_tool_manifest_store import YamlToolManifestStore
from tpa.persistence.repositories.agent_memory_repository import AgentMemoryRepository
from tpa.persistence.repositories.conversation_repository import ConversationRepository
from tpa.persistence.repositories.session_context_repository import SqliteSessionContextRepository
from tpa.persistence.repositories.trace_repository import SqliteTraceRepository
from tpa.persistence.session import SessionLocal

from bootstrap.builders.devices import build_devices
from bootstrap.builders.vision import build_vision
from bootstrap.builders.audio import build_audio_capture, build_indicator, build_mute_switch
from bootstrap.builders.inference import build_embedding, build_inference
from bootstrap.builders.lookup import build_lookup, build_lookup_health
from bootstrap.builders.reminders import build_reminders
from bootstrap.builders.safety import build_audit_sink, build_guardrails
from bootstrap.builders.skills import build_skills, register_google_tools
from bootstrap.builders.speaker_id import build_speaker_enrollment_service, build_speaker_id
from bootstrap.builders.system import build_notifier, build_system_control
from bootstrap.builders.voice import build_voice_session


def build_app_state(app: FastAPI) -> None:
    """Build every subsystem and attach it to `app.state`."""
    cfg = load_full_config()
    app.state.full_config = cfg

    event_bus = EventBus()
    app.state.event_bus = event_bus

    approval_broker = ApprovalBroker(bus=event_bus)
    app.state.approval_broker = approval_broker

    audit_sink = build_audit_sink(cfg)
    governance = build_governance(cfg.governance, audit_sink)
    app.state.governance = governance

    inference_lock = asyncio.Lock()  # shared by the main LLM and the router model: one inference at a time
    inference_thread_lock = threading.Lock()  # same guarantee inside the worker threads (survives an abandoned call)
    inference_client, inference_backend = build_inference(cfg, governance, inference_lock, inference_thread_lock)
    app.state.inference_client = inference_client
    app.state.inference_backend = inference_backend
    app.state.embedding_provider = build_embedding(cfg)

    from tpa.persistence.vector_store import SqliteVectorStore
    from service.memory.memory_indexer import MemoryIndexer
    from service.memory.semantic_recall import SemanticRecall
    vector_store = SqliteVectorStore()
    memory_indexer = MemoryIndexer(vector_store, app.state.embedding_provider)
    conversation_repo = ConversationRepository()
    app.state.conversation_repo = conversation_repo

    def _memory_date(hit):
        """The day a recalled summary or exchange is from (None for anything else, e.g. a saved fact)."""
        if hit.source == "summary":
            return conversation_repo.summary_created_at(int(hit.ref_id))
        if hit.source == "exchange":
            return conversation_repo.turn_created_at(int(hit.ref_id))
        return None

    semantic_recall = SemanticRecall(
        vector_store, app.state.embedding_provider,
        top_k=cfg.embedding.top_k, min_score=cfg.embedding.min_score,
        sources=tuple(cfg.embedding.sources), margin=cfg.embedding.margin, when=_memory_date,
        max_hit_chars=cfg.embedding.max_hit_chars, max_total_chars=cfg.embedding.max_total_chars,
        conversation_intent_min=cfg.embedding.conversation_intent_min, min_score_by_source=cfg.embedding.min_score_by_source,
    )
    app.state.vector_store = vector_store
    app.state.memory_indexer = memory_indexer
    app.state.semantic_recall = semantic_recall
    memory_repo = AgentMemoryRepository(conversation_repo=conversation_repo)
    def _index_exchange(turn_id: int, note: str) -> None:
        """Embed a finished question-and-answer in the background, off the reply path (it is a no-op without embeddings)."""
        if not memory_indexer.enabled:
            return
        try:
            asyncio.get_running_loop().create_task(memory_indexer.index("exchange", str(turn_id), note))
        except RuntimeError:
            pass   # no running loop (e.g. a test): the startup backfill will pick it up

    conversation = ConversationManager(
        repo=conversation_repo, max_history=cfg.app.max_history, summaries_limit=cfg.app.chat_summaries,
        on_exchange=_index_exchange,
    )
    # Exposed so controller/routes/health.py can actually probe the DB;
    # without this its check silently reported False forever.
    app.state.db_session_factory = SessionLocal
    async def _index_summary(summary_id: int, text: str) -> None:
        await memory_indexer.index("summary", str(summary_id), text)

    summariser = ConversationSummariser(
        repo=conversation_repo,
        client=inference_client,
        summary_model=cfg.inference.model_alias,
        summary_num_predict=cfg.inference.summary_num_predict,
        on_summary=_index_summary if memory_indexer.enabled else None,
    )
    app.state.conversation = conversation
    app.state.memory = memory_repo
    app.state.conversation_summariser = summariser

    async def _reindex_facts() -> None:
        try:
            vector_store.clear("fact")
            for i, fact in enumerate(knowledge.list_facts()):
                await memory_indexer.index("fact", str(i), fact)
        except Exception as e:
            logger.warning(f"[memory-index] fact reindex failed: {e}")

    def _reindex_facts_soon() -> None:
        try:
            asyncio.get_running_loop().create_task(_reindex_facts())
        except RuntimeError:
            pass  # no running loop (e.g. tests) — indexing is best-effort

    knowledge = KnowledgeBase(
        store=JsonKnowledgeStore(),
        on_change=_reindex_facts_soon if memory_indexer.enabled else None,
    )
    app.state.knowledge = knowledge

    from tpa.persistence.repositories.task_repository import SqliteTaskRepository
    from service.tasks.task_service import TaskService
    from service.reminders.reminder_service import ChangeSignal
    calendar_changed = ChangeSignal()   # fired when the calendar or the task list changes, so reminders re-read at once
    app.state.calendar_changed = calendar_changed
    task_service = TaskService(SqliteTaskRepository(), on_change=calendar_changed.fire)
    app.state.task_service = task_service

    hooks, permissions = build_guardrails(cfg)
    skill_registry, skill_runner = build_skills(cfg, hooks)
    from service.skills.builtin.tasks import TasksSkill
    tool_manifests = {m.name: m for m in YamlToolManifestStore().load_all()}
    online = cfg.privacy.online
    lookup_service = lookup_search = None
    if online.enabled:
        lookup_service, lookup_search = build_lookup(cfg, tool_manifests)
    else:
        tool_manifests = {n: m for n, m in tool_manifests.items() if not m.requires_online}
        logger.info("[lookup] online tools disabled (privacy.online.enabled=false)")
    room_cfg = cfg.sensing.room
    room_driver = None
    if room_cfg.enabled:
        try:
            from tpa.sensors.room_drivers import build_room_driver
            room_driver = build_room_driver(
                room_cfg.driver, gpio_pin=room_cfg.gpio_pin,
                mock_temperature_c=room_cfg.mock_temperature_c, mock_humidity_pct=room_cfg.mock_humidity_pct,
            )
        except Exception as e:   # e.g. the DHT11 library is not installed on this machine
            logger.error(f"[room] sensor driver {room_cfg.driver!r} unavailable, room feature off: {e}")
    if room_driver is None:   # no sensor: the tool leaves the control prompt entirely
        tool_manifests = {n: m for n, m in tool_manifests.items() if n != "get_room_temperature"}
    music_on = cfg.music.enabled and cfg.privacy.online.enabled
    music_off_reason = None if music_on else ("off" if not cfg.music.enabled else "offline")   # shown to the phone
    if cfg.music.enabled and not music_on:
        logger.warning("[music] enabled but privacy.online.enabled is false: music is off")
    if music_on:
        from tpa.audio.music_adapters import find_program
        absent = [f"{label} ({program!r})" for label, program in (("mpv", cfg.music.mpv_path), ("yt-dlp", cfg.music.ytdlp_path))
                  if find_program(program) is None]
        if absent:   # music turns itself off, and the tool leaves the prompt, until the setup script (or you) installs them
            logger.warning(f"[music] off: not found: {', '.join(absent)}. Run scripts/setup_pi.sh (Pi) or scripts/setup_windows.ps1 (Windows), "
                           "or install them (see config/music.yaml)")
            music_on = False
            music_off_reason = "tools_missing"
    if not music_on:   # the tool leaves the control prompt entirely
        tool_manifests = {n: m for n, m in tool_manifests.items() if n != "music_control"}
    app.state.tool_manifests = tool_manifests
    skill_registry.register(TasksSkill(
        service=task_service,
        manifest=tool_manifests["tasks"],
        permission_level=cfg.skills.tasks.permission_level,
        enabled=cfg.skills.tasks.enabled,
    ))
    from service.skills.builtin.remember import RememberSkill
    if "remember" in tool_manifests:
        skill_registry.register(RememberSkill(knowledge=knowledge, manifest=tool_manifests["remember"]))
    if lookup_service is not None:
        from service.lookup.currency_lookup import CurrencyLookup
        from service.lookup.place_resolver import PlaceResolver
        from service.lookup.search_lookup import SearchLookup
        from service.lookup.weather_lookup import WeatherLookup
        from service.skills.builtin.currency import ConvertCurrencySkill
        from service.skills.builtin.weather import GetWeatherForecastSkill, GetWeatherSkill
        from service.skills.builtin.web_search import WebSearchSkill

        places = PlaceResolver(lookup_service, default_place=online.default_place)
        weather_lookup = WeatherLookup(lookup_service, places)
        for skill in (
            GetWeatherSkill(weather_lookup, tool_manifests["get_weather"]),
            *([GetWeatherForecastSkill(weather_lookup, tool_manifests["get_weather_forecast"])]
              if "get_weather_forecast" in tool_manifests else []),
            ConvertCurrencySkill(CurrencyLookup(lookup_service), tool_manifests["convert_currency"]),
            WebSearchSkill(SearchLookup(lookup_service), tool_manifests["web_search"]),
        ):
            skill_registry.register(skill)
    app.state.google_auth, calendar_port = (
        register_google_tools(online, tool_manifests, skill_registry, calendar_changed.fire) if online.enabled else (None, None)
    )
    app.state.lookup_service = lookup_service
    app.state.lookup_health = (
        build_lookup_health(lookup_service, lookup_search, online.search_api_key_env) if lookup_service else None
    )
    app.state.hook_registry = hooks
    app.state.permission_manager = permissions
    app.state.skill_registry = skill_registry
    app.state.skill_runner = skill_runner

    system_control = build_system_control()
    from service.skills.builtin.system_control import AppControlSkill, DeviceStatusSkill, VolumeControlSkill
    for tool_name, skill_cls in (
        ('app_control', AppControlSkill), ('volume_control', VolumeControlSkill), ('device_status', DeviceStatusSkill),
    ):
        manifest = tool_manifests.get(tool_name)
        if manifest is not None:
            skill_registry.register(skill_cls(system_control, manifest, governance=governance))
    from service.sensing.room_state import RoomState
    room_state = RoomState()
    app.state.room_state = app.state.room_sensor = None
    if room_driver is not None:
        from service.sensing.room_sensor import RoomSensor
        from service.skills.builtin.room import RoomTemperatureSkill
        skill_registry.register(RoomTemperatureSkill(
            tool_manifests["get_room_temperature"], room_state, room_cfg.max_age_sec,
        ))
        app.state.room_state = room_state
        app.state.room_sensor = RoomSensor(room_driver, room_state, poll_interval=room_cfg.poll_interval_sec)
    if 'current_time' in tool_manifests:
        from service.skills.builtin.clock import CurrentTimeSkill
        skill_registry.register(CurrentTimeSkill(tool_manifests['current_time']))
    agent_registry = AgentRegistry()
    trace_repo = SqliteTraceRepository()
    app.state.trace_repo = trace_repo
    session_state = SessionStateService(
        SqliteSessionContextRepository(), tool_manifests, ttl_sec=cfg.agents.session_ttl_sec,
    )
    app.state.session_state = session_state
    guard = ToolUseGuard(tool_manifests.values())
    count_tokens = getattr(inference_backend, "count_tokens", None) or estimate_tokens
    composer = PromptComposer(
        FilePromptStore(), list(tool_manifests.values()), budgets=PromptBudgets(**cfg.prompting.budgets), count_tokens=count_tokens,
        turn_chars=cfg.prompting.turn_chars, user_message_chars=cfg.prompting.user_message_chars,
    )
    responder = ResponderAgent(
        client=inference_client, conversation=conversation, knowledge=knowledge,
        model=cfg.agents.responder.model, memory=memory_repo,
        recall=semantic_recall,
        # Chat is only reached when no tool ran, so any action claim is unbacked: it is vetted as it is generated.
        reply_veto=guard.claims_action,
        persona=FilePromptStore().get("persona_chat") if cfg.app.chat_persona == "compact" else None,
        chat_budget_tokens=PromptBudgets(**cfg.prompting.budgets).chat, history_turn_chars=cfg.prompting.turn_chars,
    )
    agent_registry.register(responder)
    orchestrator = AssistantOrchestrator(
        decoder=ControlDecoder(
            client=inference_client, composer=composer, manifests=tool_manifests,
            model=cfg.agents.responder.model, exchanges=cfg.prompting.control_history_exchanges,
        ),
        composer=composer, skill_runner=skill_runner, manifests=tool_manifests, responder=responder,
        client=inference_client, conversation=conversation, claims_action=guard.claims_action,
        session_state=session_state, traces=trace_repo, memory=memory_repo, model=cfg.agents.responder.model,
        conversation_intent=semantic_recall.conversation_intent_score if cfg.embedding.conversation_intent_min is not None else None,
        conversation_intent_min=cfg.embedding.conversation_intent_min or 0.82,
        conversation_intent_soft_min=cfg.embedding.conversation_intent_soft_min,
        adaptive_reply=cfg.app.adaptive_reply, mixed_intent=cfg.app.mixed_intent,
        parallel_tool_calls=cfg.app.parallel_tool_calls,
        room_context=(lambda: room_line(room_state.get(), datetime.now(), room_cfg.max_age_sec)) if room_driver is not None else None,
        multilingual=cfg.app.multilingual.enabled,
        reject_unknown_language=cfg.app.multilingual.reject_unknown,
        unknown_language_reply=cfg.app.multilingual.unknown_reply,
        min_language_letters=cfg.app.multilingual.min_letters,
    )
    app.state.orchestrator = orchestrator
    supervisor = SupervisorAgent(model=cfg.agents.supervisor.model, orchestrator=orchestrator)
    agent_registry.register(supervisor)
    app.state.agent_registry = agent_registry
    app.state.supervisor = supervisor
    app.state.ambient_dispatcher = AmbientDispatcher(
        debounce_window_sec=cfg.sensing.debounce_window_sec, rate_limit_max=cfg.sensing.rate_limit_max,
        rate_limit_window_sec=cfg.sensing.rate_limit_window_sec, proactivity=cfg.sensing.proactivity,
        conversation=conversation,
    )

    app.state.notifier = build_notifier()
    app.state.egress_allow_list = frozenset(cfg.privacy.online.allowlist)

    # Privacy chain (REQ-M-04/M-05). CaptureGate owns mute state and is the
    # only thing allowed to start/stop the mic or drive the indicator.
    # The same capture adapter instance is shared with the voice loop —
    # two independent mic streams would fight over the device.
    mute_switch = build_mute_switch(cfg)
    audio_capture = build_audio_capture(cfg)
    capture_gate = CaptureGate(
        mute_switch=mute_switch,
        indicator=build_indicator(),
        audio=audio_capture,
        bus=event_bus,
    )
    app.state.mute_switch = mute_switch
    app.state.capture_gate = capture_gate

    # Multi-speaker recognition (REQ-M-04 extension: "don't react to stray
    # talk" sharpened to "don't react to an unenrolled voice"). Built once
    # here so the live recognizer instance is reachable both from the voice
    # loop and from the enrollment route's reload_profiles() call.
    speaker_recognizer = build_speaker_id(cfg)
    app.state.speaker_recognizer = speaker_recognizer
    app.state.speaker_enrollment_service = build_speaker_enrollment_service(cfg, audio_capture)

    # Always-on voice loop. Gated by CaptureGate, so building it does not
    # mean the mic is open — it opens only when the gate says unmuted.
    app.state.voice_session = build_voice_session(
        cfg, audio=audio_capture, capture_gate=capture_gate,
        supervisor=supervisor, event_bus=event_bus, speaker_id=speaker_recognizer,
    )

    app.state.music = None
    app.state.music_off_reason = music_off_reason   # None while music is on; otherwise off | offline | tools_missing
    if music_on:
        from service.music.music_service import MusicService
        from service.skills.builtin.music import MusicSkill
        from tpa.audio.music_adapters import MpvPlayer, YtDlpResolver
        mc = cfg.music
        voice = app.state.voice_session
        music = MusicService(
            YtDlpResolver(mc.ytdlp_path, mc.search_results, mc.search_timeout_sec), MpvPlayer(mc.mpv_path, mc.ytdlp_path),
            capture_gate, max_seconds=mc.max_minutes * 60, is_busy=voice.is_busy if voice is not None else None,
        )
        capture_gate.add_mute_listener(music.on_mute_change)   # the mute switch and /mute /unmute are the guaranteed stop
        skill_registry.register(MusicSkill(tool_manifests["music_control"], music))
        app.state.music = music
        logger.info(f"[music] on: songs play through mpv for at most {mc.max_minutes:g} minute(s); the mic is closed meanwhile")

    app.state.reminders = build_reminders(
        cfg, calendar=calendar_port, tasks=task_service, voice_session=app.state.voice_session,
        calendar_changed=calendar_changed, music=app.state.music,
    )

    app.state.device_service = build_devices(cfg)   # phone pairing/auth; independent of every subsystem above
    app.state.vision = build_vision(cfg)   # phone camera -> local vision model; None unless vision.enabled; touches nothing above

    logger.info(f"Bootstrap: {agent_registry.count} agents, {len(skill_registry.list_all())} skills registered")
