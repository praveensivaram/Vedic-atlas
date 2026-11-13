"""calendar_create, current_time, sent-mail search, and the guards that keep calendar requests out of tasks and
pronoun-only follow-ups out of the web search. Model-free: every value the model would choose is passed in directly."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from domain.entities.agent_context import AgentContext
from domain.entities.calendar_event import CalendarEvent
from domain.entities.mail_message import MailMessage
from domain.policies import calendar_policy as cal
from domain.policies import mail_policy as mail
from exceptions.exception import ToolUnavailableError
from service.lookup.search_lookup import _first_sentences
from service.skills.builtin.calendar_create import CalendarCreateSkill
from service.skills.builtin.clock import CurrentTimeSkill
from service.skills.builtin.gmail import GmailSearchSkill
from service.skills.builtin.web_search import WebSearchSkill
from tpa.filestore.yaml_tool_manifest_store import YamlToolManifestStore
from tpa.online.google.calendar_client import GoogleCalendarClient
from tpa.online.google.gmail_parser import to_message

MANIFESTS = {m.name: m for m in YamlToolManifestStore().load_all()}
NOW = datetime(2026, 10, 8, 14, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))   # a Thursday


def run(coro):
    return asyncio.run(coro)


def ctx(message="add the software review and the dentist and the review at 5 pm"):
    return AgentContext(user_message=message)


class FakeCalendar:
    def __init__(self, existing=(), create_error=None):
        self.existing, self.create_error, self.created = list(existing), create_error, []

    async def list_events(self, start, end, limit=20):
        return [e for e in self.existing if start <= e.start < end]

    async def create_event(self, title, start, end):
        if self.create_error:
            raise self.create_error
        self.created.append((title, start, end))
        return CalendarEvent("new", title, start, end)


# ---- calendar_create -----------------------------------------------------------------------------------------------

def skill(cal_port):
    return CalendarCreateSkill(cal_port, MANIFESTS["calendar_create"], now=lambda: NOW)


def test_creates_the_event_tomorrow_at_five_and_reads_it_back_from_what_was_stored():
    fc = FakeCalendar()
    res = run(skill(fc).execute(ctx(), title="Software review", date_offset=1, time="17:00"))
    assert res.success and res.metadata["spoken"] == "Added Software review to your calendar tomorrow at 5 PM."
    (title, start, end), = fc.created
    assert start == NOW.replace(day=9, hour=17, minute=0) and end - start == timedelta(hours=1)


def test_duration_is_honoured_and_defaults_to_one_hour():
    fc = FakeCalendar()
    run(skill(fc).execute(ctx(), title="Dentist", time="16:45", duration_minutes=30))
    (_, start, end), = fc.created
    assert end - start == timedelta(minutes=30) and start.day == 8


def test_the_same_event_is_not_added_twice():
    existing = CalendarEvent("1", "Software review", NOW.replace(day=9, hour=17, minute=0))
    fc = FakeCalendar([existing])
    res = run(skill(fc).execute(ctx(), title="software review", date_offset=1, time="17:00"))
    assert res.success and fc.created == [] and "already on your calendar tomorrow at 5 PM" in res.metadata["spoken"]


@pytest.mark.parametrize("args,sentence", [
    ({"title": "", "time": "17:00"}, "What should the event be called?"),
    ({"title": "Review", "time": ""}, "What time should I set it for?"),
    ({"title": "Review", "time": "25:00"}, "What time should I set it for?"),
    ({"title": "Review", "time": "17:00", "date_offset": 9}, "I can only add events up to 6 days ahead."),
    ({"title": "Review", "time": "09:00", "date_offset": 0}, "That time has already passed."),
    ({"title": "Review", "time": "17:00", "duration_minutes": 1}, "An event can last between 5 minutes and 8 hours."),
])
def test_bad_input_is_refused_in_plain_speech_and_nothing_is_stored(args, sentence):
    fc = FakeCalendar()
    res = run(skill(fc).execute(ctx(), **args))
    assert not res.success and res.metadata["spoken"] == sentence and fc.created == []


def test_a_sign_in_without_calendar_write_access_says_so_and_how_to_fix_it():
    fc = FakeCalendar(create_error=ToolUnavailableError("c", "calendar", "403", status=403))
    res = run(skill(fc).execute(ctx(), title="Review", time="17:00", date_offset=1))
    assert not res.success and "permission to add calendar events" in res.metadata["spoken"]


def test_the_manifest_cannot_invite_anyone_and_keeps_content_private():
    m = MANIFESTS["calendar_create"]
    assert m.private and m.reply_mode == "template" and {p.name for p in m.params} == {"title", "date_offset", "time", "duration_minutes"}


def test_google_adapter_posts_the_event_without_attendees():
    class Http:
        def __init__(self):
            self.sent = []

        async def post_json(self, url, *, category, json, headers=None):
            self.sent.append(json)
            return {"id": "e1", "summary": json["summary"], "start": json["start"], "end": json["end"]}

    class Auth:
        async def headers(self):
            return {"Authorization": "Bearer T"}

    http = Http()
    event = run(GoogleCalendarClient(http, Auth()).create_event("Review", NOW, NOW + timedelta(hours=1)))
    assert event.title == "Review" and "attendees" not in http.sent[0] and http.sent[0]["start"]["dateTime"] == NOW.isoformat()


# ---- follow-ups and relative times (the 2026-10-27 "sleep" exchange) ---------------------------------------------------

LATE = datetime(2026, 10, 27, 22, 21, 46, tzinfo=NOW.tzinfo)
SLEEP = "can you create event that i need to sleep in my calendar in next 10 mins?"


def late_skill(fc, now=LATE):
    return CalendarCreateSkill(fc, MANIFESTS["calendar_create"], now=lambda: now)


def asked(question, at=LATE):
    """The conversation so far: the sleep request, then the tool's own question, said at `at` (stored naive, local)."""
    return [
        {"role": "user", "content": SLEEP, "created_at": at.replace(tzinfo=None)},
        {"role": "assistant", "content": question, "created_at": at.astimezone().replace(tzinfo=None)},
    ]


def test_in_next_ten_minutes_is_worked_out_from_the_clock_not_the_models_guess():
    fc = FakeCalendar()
    res = run(late_skill(fc).execute(ctx(SLEEP), title="Sleep", date_offset=0, time="22:21", duration_minutes=10))
    assert res.success, res.metadata["spoken"]
    (title, start, end), = fc.created
    assert title == "Sleep" and (start.hour, start.minute) == (22, 31) and end - start == timedelta(hours=1)


@pytest.mark.parametrize("phrase,minutes", [
    ("in 10 mins", 10), ("in the next 15 minutes", 15), ("in an hour", 60), ("in half an hour", 30), ("in two hours", 120),
])
def test_relative_starts(phrase, minutes):
    start = cal.relative_start(f"add yoga {phrase}", LATE)
    assert start == (LATE + timedelta(minutes=minutes)).replace(second=0)


def test_a_relative_start_past_midnight_lands_tomorrow():
    fc = FakeCalendar()
    res = run(late_skill(fc).execute(ctx("add a call with Ravi in 2 hours"), title="Call with Ravi", time="22:21"))
    assert res.success and fc.created[0][1].day == 28 and fc.created[0][1].hour == 0
    assert res.metadata["spoken"].startswith("Added Call with Ravi to your calendar tomorrow")


def test_a_stated_length_is_kept_beside_a_relative_start():
    fc = FakeCalendar()
    run(late_skill(fc).execute(ctx("add a nap in 10 mins for 20 minutes"), title="Nap", duration_minutes=20))
    (_, start, end), = fc.created
    assert end - start == timedelta(minutes=20)


def test_answering_the_time_question_keeps_the_title_from_the_request():
    fc = FakeCalendar()
    c = AgentContext(user_message="10:30 PM", conversation_history=asked(cal.ASK_TIME))
    res = run(late_skill(fc, LATE + timedelta(seconds=40)).execute(c, title="Sleep", date_offset=0, time="22:30"))
    assert res.success, res.metadata["spoken"]
    assert fc.created[0][0] == "Sleep" and (fc.created[0][1].hour, fc.created[0][1].minute) == (22, 30)


def test_answering_the_title_question_keeps_the_time_from_the_request():
    fc = FakeCalendar()
    history = [
        {"role": "user", "content": "put something on my calendar at 11 pm", "created_at": LATE.replace(tzinfo=None)},
        {"role": "assistant", "content": cal.ASK_TITLE, "created_at": LATE.astimezone().replace(tzinfo=None)},
    ]
    c = AgentContext(user_message="call it reading", conversation_history=history)
    res = run(late_skill(fc).execute(c, title="Reading", time="23:00"))
    assert res.success and fc.created[0][0] == "Reading"


def test_an_old_question_does_not_turn_small_talk_into_a_booking():
    fc = FakeCalendar()
    c = AgentContext(user_message="hey veda, how are you doing?", conversation_history=asked(cal.ASK_TITLE))
    res = run(late_skill(fc, LATE + timedelta(minutes=18)).execute(c, title="Sleep", date_offset=0, time="22:30"))
    assert not res.success and fc.created == []
    assert res.metadata["spoken"] == "I didn't add anything to your calendar."


def test_without_a_question_just_asked_the_answer_alone_is_not_enough():
    fc = FakeCalendar()
    history = [{"role": "user", "content": SLEEP, "created_at": LATE.replace(tzinfo=None)},
               {"role": "assistant", "content": "Nothing on your calendar today.", "created_at": LATE.replace(tzinfo=None)}]
    c = AgentContext(user_message="10:30 PM", conversation_history=history)
    res = run(late_skill(fc).execute(c, title="Sleep", time="22:30"))
    assert not res.success and res.metadata["spoken"] == cal.ASK_TITLE and fc.created == []


# ---- tasks must not swallow calendar entries -------------------------------------------------------------------------


# ---- current_time ----------------------------------------------------------------------------------------------------

def test_current_time_and_date_come_from_the_clock():
    sk = CurrentTimeSkill(MANIFESTS["current_time"], now=lambda: NOW)
    assert run(sk.execute(ctx())).metadata["spoken"] == "It's 2:30 PM."
    assert run(sk.execute(ctx(), what="time")).metadata["spoken"] == "It's 2:30 PM."
    assert run(sk.execute(ctx(), what="date")).metadata["spoken"] == "Today is Thursday 8 October 2026."


# ---- sent mail --------------------------------------------------------------------------------------------------------

class FakeMail:
    def __init__(self, found=()):
        self.found, self.searches = list(found), []

    async def search(self, query, limit=5):
        self.searches.append(query)
        return self.found


def sent_msg():
    return MailMessage(
        id="m1", sender_name="Me", sender_address="me@x.com", subject="Tomorrow's Calendar Invite",
        recipients=("manoj@example.com",), recipient_names=("Manoj Routhu",),
    )


def test_sent_folder_searches_in_sent_whatever_query_the_model_wrote():
    fm = FakeMail([sent_msg()])
    sk = GmailSearchSkill(fm, MANIFESTS["gmail_search"])
    res = run(sk.execute(ctx(), folder="sent", query="to:manoj@example.com"))
    assert fm.searches == ["in:sent to:manoj@example.com"]
    assert res.metadata["spoken"] == "1 sent email. To Manoj Routhu: Tomorrow's Calendar Invite."
    run(sk.execute(ctx(), folder="sent"))
    assert fm.searches[-1] == "in:sent"


def test_existing_inbox_behaviour_is_unchanged_when_no_folder_is_given():
    fm = FakeMail([])
    sk = GmailSearchSkill(fm, MANIFESTS["gmail_search"])
    run(sk.execute(ctx(), unread_only=True))
    run(sk.execute(ctx(), query="from:priya"))
    run(sk.execute(ctx(), folder="inbox", query="invoice"))
    assert fm.searches == ["is:unread in:inbox", "from:priya", "in:inbox invoice"]


def test_nothing_sent_is_said_as_such():
    assert mail.spoken_list([], sent=True) == "You haven't sent any matching emails."


def test_recipient_names_are_parsed_from_the_to_header():
    item = {"id": "1", "payload": {"headers": [
        {"name": "From", "value": "me@x.com"}, {"name": "To", "value": "Manoj Routhu <m@x.com>, plain@y.com"},
        {"name": "Subject", "value": "Hi"},
    ]}}
    m = to_message(item)
    assert m.recipients == ("m@x.com", "plain@y.com") and m.recipient_names == ("Manoj Routhu", "")
    assert m.recipient_label() == "Manoj Routhu"


# ---- web search follow-ups ----------------------------------------------------------------------------------------------


def test_initials_and_titles_do_not_end_a_spoken_sentence():
    text = "Infosys was founded by N. R. Narayana Murthy, K. Dinesh and Ashok Arora in Pune. More follows here."
    assert _first_sentences(text, 90) == "Infosys was founded by N. R. Narayana Murthy, K. Dinesh and Ashok Arora in Pune."
    assert _first_sentences("He met Dr. Rao at St. Mary. Second.", 40) == "He met Dr. Rao at St. Mary. Second."
