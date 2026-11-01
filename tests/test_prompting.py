"""Phase 1: tool manifests and PromptComposer budgets.

No model, no server. Run: pytest tests/test_prompting.py
"""

from datetime import datetime

import pytest

from core.enums import ExceptionCode
from domain.entities.conversation import Turn
from domain.policies.token_budget_policy import PromptBudgets, estimate_tokens
from exceptions.exception import AppException
from service.prompting.prompt_composer import PromptComposer
from tpa.filestore.file_prompt_store import FilePromptStore
from tpa.filestore.yaml_tool_manifest_store import YamlToolManifestStore

_NOW = datetime(2026, 10, 3, 19, 40)


def _manifests():
    return YamlToolManifestStore().load_all()


def _composer(budgets=None, count=estimate_tokens):
    return PromptComposer(FilePromptStore(), _manifests(), budgets=budgets, count_tokens=count, now=lambda: _NOW)


def _turn(role, text):
    return Turn(id=None, session_id="s", role=role, content=text, created_at=_NOW)


# ---- manifests ---------------------------------------------------------

def test_real_manifests_load_and_are_valid():
    by_name = {m.name: m for m in _manifests()}
    tasks = by_name["tasks"]
    assert tasks.agent == "tasks" and tasks.reply_mode == "template"
    assert {p.name for p in tasks.params} >= {"action", "title"}
    assert "task_id" not in {p.name for p in tasks.params}   # the model never sees real ids, so it must not be able to emit one
    assert tasks.examples and tasks.triggers and tasks.claims


def test_invalid_manifest_fails_loudly(tmp_path):
    (tmp_path / "bad.yaml").write_text("name: Bad Name\nagent: x\ndescription: d\n", encoding="utf-8")
    with pytest.raises(AppException) as err:
        YamlToolManifestStore(tmp_path).load_all()
    assert err.value.code == ExceptionCode.CONFIG_ERROR


def test_manifest_rejects_long_description_and_foreign_example(tmp_path):
    long_desc = " ".join(["word"] * 40)
    (tmp_path / "a.yaml").write_text(f"name: a\nagent: x\ndescription: {long_desc}\n", encoding="utf-8")
    with pytest.raises(AppException):
        YamlToolManifestStore(tmp_path).load_all()
    (tmp_path / "a.yaml").write_text(
        "name: a\nagent: x\ndescription: ok\nparams: {q: {type: string}}\n"
        "examples:\n  - user: hi\n    calls: [{tool: other, args: {q: x}}]\n",
        encoding="utf-8",
    )
    with pytest.raises(AppException):
        YamlToolManifestStore(tmp_path).load_all()


def test_missing_prompt_raises_config_error(tmp_path):
    with pytest.raises(AppException) as err:
        FilePromptStore(tmp_path).get("nope")
    assert err.value.code == ExceptionCode.CONFIG_ERROR


# ---- composer -----------------------------------------------------------

def test_narrate_stage_caps_tool_result():
    huge = "result line " * 600
    p = _composer().narrate_stage("what's the news", [huge])
    assert "tool_result" in p.trimmed
    assert p.tokens <= PromptBudgets().narrate
    assert "QUESTION: what's the news" in p.prompt and "Answer:" in p.prompt


def test_default_prompts_are_small():
    c = _composer()
    assert c.control_stage("hi").tokens < PromptBudgets().control
    assert c.narrate_stage("hi", ["ok"]).tokens < 300


# ---- content stage (data plane) -----------------------------------------------------------------------

def test_content_tasks_share_one_cached_prefix_and_differ_only_in_the_tail():
    c = _composer()
    prompts = {t: c.content_stage(t, "the document body", "what does it say") for t in ("narrate", "summarise", "draft_reply", "extract")}
    assert len({p.system for p in prompts.values()}) == 1
    assert len({p.prompt for p in prompts.values()}) == 4
    for task, p in prompts.items():
        assert p.prompt.startswith("TASK: ") and "the document body" in p.prompt and "Answer:" in p.prompt


def test_content_stage_has_no_tools_no_history_and_no_volatile_text_in_system():
    system = _composer().content_stage("summarise", "doc", "q").system
    for forbidden in ("get_weather", "convert_currency", "tasks", "ACTIVE", "RECENT", "TODAY", "QUESTION"):
        assert forbidden not in system


def test_narrate_stage_is_the_narrate_task():
    c = _composer()
    assert c.narrate_stage("q", ["a", "b"]) == c.content_stage("narrate", "a\nb", "q")
    assert "TOOL RESULT:" in c.narrate_stage("q", ["a"]).prompt


def test_the_tools_own_cap_is_applied_before_composition():
    doc = "word " * 2000
    capped = _composer().content_stage("summarise", doc, "q", max_tokens=40)
    uncapped = _composer().content_stage("summarise", doc, "q")
    assert "tool_result" in capped.trimmed and capped.sections["tool_result"] <= 42
    assert capped.sections["tool_result"] < uncapped.sections["tool_result"]
    assert capped.tokens <= PromptBudgets().narrate


def test_unknown_content_task_is_a_programming_error():
    import pytest
    with pytest.raises(ValueError):
        _composer().content_stage("summarize", "doc", "q")
