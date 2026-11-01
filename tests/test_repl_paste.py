"""The verbatim paste reader: no line editing, so complex scripts survive."""

from io import StringIO

from controller.cli.paste import ensure_utf8_streams, read_message

HINDI = "\u0906\u091c \u092e\u0947\u0930\u093e \u091c\u0928\u094d\u092e\u0926\u093f\u0928 \u0939\u0948"
TELUGU = "\u0c28\u0c3e\u0c15\u0c41 \u0c2a\u0c41\u0c38\u0c4d\u0c24\u0c15\u0c3e\u0c32\u0c41 \u0c1a\u0c26\u0c35\u0c21\u0c02 \u0c07\u0c37\u0c4d\u0c1f\u0c02"


def test_reads_unicode_verbatim_until_the_terminator():
    stream = StringIO(f"{HINDI}\n{TELUGU}\n.\nignored after terminator\n")
    assert read_message(stream) == f"{HINDI}\n{TELUGU}"


def test_crlf_and_surrounding_whitespace_are_normalised():
    stream = StringIO(f"  {HINDI}  \r\n.\r\n")
    assert read_message(stream) == HINDI


def test_eof_without_a_terminator_keeps_what_was_read():
    stream = StringIO(f"{HINDI}\n{TELUGU}\n")
    assert read_message(stream) == f"{HINDI}\n{TELUGU}"


def test_empty_paste_is_empty():
    assert read_message(StringIO(".\n")) == ""
    assert read_message(StringIO("")) == ""


def test_utf8_hardening_tolerates_streams_that_cannot_be_reconfigured():
    ensure_utf8_streams()  # StringIO-style streams have no reconfigure; must not raise
