"""The localisation stage and the echo filter."""

from domain.entities.agent_context import AgentContext
from domain.policies.echo_policy import fallback_reply, is_echo
from service.agent.responder import ResponderAgent
from service.prompting.prompt_composer import PromptComposer
from tpa.filestore.file_prompt_store import FilePromptStore
from tpa.filestore.yaml_tool_manifest_store import YamlToolManifestStore

HINDI = "\u0906\u091c \u092e\u0947\u0930\u093e \u091c\u0928\u094d\u092e\u0926\u093f\u0928 \u0939\u0948"
HINDI_ASK = "\u0915\u094d\u092f\u093e \u0924\u0941\u092e \u092e\u0941\u0902\u092c\u0908 \u092e\u0947\u0902 \u092a\u093e\u0930\u094d\u091f\u0940 \u0915\u0930\u0928\u0947 \u0915\u0947 \u0932\u093f\u090f \u091c\u0917\u0939 \u0922\u0942\u0902\u0922"
HINDI_DONE = HINDI_ASK + " \u0938\u0915\u0924\u0947 \u0939\u094b?"


def _composer():
    return PromptComposer(FilePromptStore(), YamlToolManifestStore().load_all())


# ---- echo filter -------------------------------------------------------------

def test_exact_repeat_is_an_echo():
    assert is_echo(HINDI, HINDI)
    assert is_echo("hello there", "hello there")


def test_punctuation_and_case_do_not_hide_an_echo():
    assert is_echo(HINDI + "?", HINDI)
    assert is_echo("Hello There!", "hello there")


def test_a_completed_fragment_counts_as_an_echo():
    assert is_echo(HINDI_DONE, HINDI_ASK)


def test_a_real_answer_is_not_an_echo():
    assert not is_echo("Jazz am Abend ist etwas Feines, Anna.", "Hallo, ich bin Anna. Ich höre abends gern Jazz.")
    assert not is_echo("The weather is 27 degrees in Chennai.", HINDI_ASK)


def test_short_replies_and_empty_text_are_not_echoes():
    assert not is_echo("ok", "a much longer message from the user")
    assert not is_echo("", HINDI)
    assert not is_echo(HINDI, "")


# ---- translate prompt --------------------------------------------------------

def test_translate_stage_carries_the_sentence_and_no_user_text():
    stage = _composer().translate_stage("In Chennai it's 27 degrees.", "Hindi")
    assert "TARGET LANGUAGE: Hindi" in stage.prompt
    assert "TEXT:\nIn Chennai it's 27 degrees." in stage.prompt
    assert "QUESTION" not in stage.prompt and "USER:" not in stage.prompt
    assert "Translate TEXT into TARGET LANGUAGE" in stage.system
    assert "never repeat the user's question" in stage.system


# ---- fixed fallback + retry prompt ------------------------------------------

def test_fallback_reply_is_controlled_and_in_the_turn_language():
    assert fallback_reply(None) == fallback_reply("English")
    assert fallback_reply("Klingon") == fallback_reply("English")
    assert "\u091c\u0935\u093e\u092c" in fallback_reply("hindi")     # "answer" in Hindi
    assert fallback_reply("Telugu") != fallback_reply("English")


class _Conversation:
    turns = []
    def get_summaries_block(self):
        return ""


class _Knowledge:
    def get_context(self):
        return ""


def test_the_retry_prompt_forbids_repeating_the_message():
    responder = ResponderAgent(client=None, conversation=_Conversation(), knowledge=_Knowledge())
    prompt = responder.build_prompt(AgentContext(user_message="hello there", metadata={"no_echo": True}))
    assert "previous attempt repeated" in prompt
    assert "never repeat or restate" in prompt
