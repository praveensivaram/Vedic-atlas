# Remember Less, Answer Better: Designing Memory for a Small Model on Low-Power Hardware

## On a CPU-only assistant, the prompt costs more than storage. So the most important thing memory can do is stay quiet.

Most writing about memory for AI assistants assumes storage is the expensive part and the prompt is cheap. You extract
facts with a model, embed them, retrieve the top few, and add them to every request. On a hosted model that is a
reasonable trade.

I build an offline voice assistant for low-power hardware: a few CPU cores, a few gigabytes of RAM, no GPU and no cloud
model. The language model is a quantized Qwen3 (4B, with 1.7B and 0.6B configurations for speed), run in-process with
llama-cpp-python. Speech-to-text, text-to-speech, the wake word and the embedding model all share the same CPU.

The measurements here come from a laptop CPU (an Intel Core i5-13420H, no GPU used) with the settings low-power hardware
needs: a 4-bit model, a 3,072-token context, a trimmed persona and short histories. Nothing in the memory layer needs
more than a small single-board computer such as a Raspberry Pi 5 provides: the embedding model is about 130 MB on disk
and runs on two threads, the store is a single SQLite file, and it adds no model calls.

On hardware like this the trade reverses. Storage is nearly free; the prompt is where the time goes. Every token the
model reads before it answers is time the user spends waiting, and every model call that is not the answer is a call the
user waits behind.

This article describes the memory layer I built, the measurements that shaped it, and what is still unsolved. The short
version: the most important decision a small assistant's memory makes is not what to remember, but when to stay quiet.

## The constraint

Two costs dominate a turn on a CPU-only device.

**Prompt reading ("prefill").** Before the model writes a word, it has to process every token of its prompt, and the
cost grows with the prompt's length. llama-cpp-python can keep the processed state of a prompt's beginning and reuse it
when the next prompt starts with the same tokens. On the laptop, that cache took the chat prompt from 14 seconds to 2.4.

That cache decides where memory goes. The chat prompt runs from least to most changeable: persona, saved facts, any
recalled memory, the recent conversation, then the time and the question. Facts rarely change, so they usually stay
inside the cached beginning. Recalled memory changes from turn to turn, so the cache stops where it starts, and
everything after that point, memory included, is processed again in full.

**Model calls.** A small model on a CPU takes seconds per call, and every turn already needs at least one: the
assistant's decision step chooses between answering and calling a tool before anything else happens. A memory design
that adds a call per message (to extract facts, decide what to save, or rewrite the query) adds that delay to every turn,
or competes with the next turn for the one model the device can hold.

So I set two rules before writing any memory code:

1. **Saving a memory must not need a model call of its own.**
2. **Recalled memory has a fixed budget in the prompt, and the default amount is zero.**

(Rule 2 holds in the low-compute configuration. The larger base configuration still adds the three most recent summaries
to every chat prompt, about 450 tokens; it predates these rules.)

## Four kinds of memory, four lifetimes

- **Saved facts**: things the user asks the assistant to remember ("my wife's name is …"). Always in the prompt, as a
  short list.
- **Current session**: the turns of the conversation happening now. The last few turns, trimmed to a token budget.
- **Session summaries**: one short memo per finished conversation. In the prompt only when recall finds it relevant, or
  when the user asks what we've been talking about.
- **Exchange notes**: every question and its answer, as said. In the prompt only when recall finds it relevant.

All four feed the chat reply. The decision step that picks tools does not see them; it gets only the last two exchanges
and a short note of which tool was just used, which expires after 15 minutes. That routing step is not my design and is
out of scope here.

A **session** ends after 30 minutes of silence. That one rule removes a lot of confusion: a new conversation starts with
an empty history instead of dragging in whatever was said that morning, and older conversations reach the model only
through recall. Within a session, the chat prompt shows the last eight turns, each cut to 200 characters, and drops the
oldest first to fit a 1,500-token chat budget.

**Saved facts** are stored as `topic: value`. Topic names are normalized ("fav food" and "favorite food" are the same
topic), and a new value for a topic replaces the old one. The list is capped at 40, and anything that looks like a
password, PIN or card number is refused. Facts are written by the model, but not with an extra call: when the user says
"remember that…", the decision step picks the `remember` tool and fills in the topic and value as part of its normal
decision.

**Summaries** are the only memory that needs a model call of its own. When a conversation has been over for 30 minutes
(or has built up 50 unsummarized turns and paused for two minutes), a background job asks the model for a one-paragraph
memo of at most 600 characters. It starts only after the user has gone quiet, but it uses the same model as everything
else and cannot be interrupted: if the user speaks while a summary is being written, the reply waits.

## The default amount of memory is zero

This is the part that changed my mind about how memory should work.

The usual retrieval pattern returns the top few matches and adds them to the prompt. But a vector search always returns
*something*: there is always a nearest neighbor. For an assistant that is mostly asked "what is 2 plus 2" or "set a
timer", that means old conversations about unrelated things land in the prompt on almost every turn. That costs
prompt-reading time, and it costs accuracy: irrelevant context measurably degrades language-model answers, even on
grade-school math problems ([Shi et al., ICML 2023](https://arxiv.org/abs/2302.00093)).

I measured how often it happened. With the similarity floor I started with (0.5), **15 of 16 ordinary questions** pulled
old memories into the prompt. None of them needed any.

The embedding model is BGE-small. Its cosine similarity scores (how close two texts are in meaning, from 0 to 1) turned
out to be compressed into a narrow band, roughly 0.45 to 0.85. Within that band, on my data, questions that really were
about the past scored 0.66 or higher against the right summary, and ordinary questions had a median of 0.56. A floor of
0.62 sits between the two. With it, the number of ordinary questions that got memory dropped to **4 of 16**.

Three more rules came out of the same measurements.

**A stricter floor for short notes.** Exchange notes (described below) are short, and short texts score higher against a
query than long summaries do. At the summary floor, 10 of 16 ordinary questions pulled in an old exchange. Exchanges got
their own floor of 0.70: questions about a specific detail scored at or above it, ordinary questions at or below it, and
questions repeated from an earlier conversation scored 0.8 and higher, which is a correct match. The two sources are
searched separately and then merged, so a crowd of short notes scoring just under their floor cannot push the summaries
out of the results.

**Look past rank 1, but only at near-ties.** The right summary was the top result for only 9 of 12 questions about the
past; it was often second, a few hundredths behind. Moving from "use the top result" to "keep the best hit plus every hit
within 0.05 of it, up to four", together with the 0.62 floor, put the right summary in the prompt for 10 of 12. Looking
past rank 1 is what finds the extra summary. The 0.05 margin does the opposite job: it stops the other slots filling
with weaker matches.

**A hard size limit.** Each recalled item is cut to 280 characters and the whole block to about 800, roughly 200 tokens.
The budget goes to the best matches first; a smaller, lower-ranked match can still fit when a larger one does not. What
survives is shown newest first.

After each source has been searched, the filtering is short enough to show whole:

```python
hits = [h for h in hits if h.score >= floor_for(h.source)]   # 0.62 summaries, 0.70 exchanges
if hits:
    best = hits[0].score
    hits = [h for h in hits if h.score >= best - 0.05]         # near-ties of the best only
return hits[:4]                                                # then capped to ~800 characters
```

None of this is a new algorithm. Similarity floors and score margins are old ideas in information retrieval. What I did
not expect was how much they mattered, and that the right values could only come from the actual score distribution of
the actual embedding model on real data.

## Saving without a model: exchange notes

Summaries have two problems. They only exist once a conversation is over and the model has had time to write one. And a
one-paragraph summary of a long conversation loses exactly the details people ask about later: the restaurant name, the
number, the date. Meanwhile, the chat prompt shows only the last few turns, so something said twenty minutes ago in the
same conversation is already out of view.

A common fix is to have a model extract facts from each exchange. Mem0, for example, runs an LLM extraction step on every
new message pair and then a second step to reconcile the results with existing memories
([Chhikara et al., 2025](https://arxiv.org/abs/2504.19413)). On a hosted model that is affordable. Here it is a model call
per turn, which breaks rule 1.

Instead, every question and answer is saved as a short note built from the words that were actually said:

```python
def exchange_note(user, reply, user_chars=110, reply_chars=170):
    u, r = flatten(user), flatten(reply)
    if len(u) < 6 or len(r) < 6:          # "ok", "hi": nothing worth recalling
        return ""
    return f"You: {clip(u, user_chars)} | Assistant: {clip(r, reply_chars)}"
```

The note is embedded in the background as soon as the turn is saved. The reply never waits for it, though on a busy CPU
the embedding can briefly compete with text-to-speech. No model paraphrases the note, so it cannot misquote what was
said. It cannot correct it either: if the assistant's answer was wrong, the note keeps the wrong answer and can bring it
back later.

On a copy of my conversation data, with 14 questions about specific details from earlier conversations, the answer
reached the prompt for 13 with exchange notes and summaries together, against 12 with summaries alone. That is one
question out of 14, measured at the retrieval step on conversations that already had summaries. The case that motivated
exchange notes, details from a conversation that has not been summarized yet, is not measured.

The embedding model runs with onnxruntime, limited to two threads so it leaves the rest of the CPU to the language model.
Vectors are stored as float32 blobs in the same SQLite database as the conversation history, and search scores them with
a NumPy matrix product, once per source. There is no vector database. Raw turns are deleted after 30 days, so one
person's notes stay in the thousands, not millions, and a full scan is simple and one less service to run.

Without the embedding model, recall, exchange notes and the conversation check described next are switched off, and the
assistant answers from saved facts and the recent turns alone.

## Memory needs a sense of time

Two problems had nothing to do with similarity.

**"When was that?"** "You said you'd call the dentist" means something different if it was yesterday or last month.
Every recalled item is labeled with its day ("Thu 8 Oct") and shown newest first. One gap remains: the chat prompt gives
the weekday and time but not today's date, which the model needs to relate a label like "Thu 1 Oct" to today.

**"What were we talking about?"** This question defeated similarity search. It is about the conversation itself and
shares no topic words with any summary, so it matched whatever happened to be closest, or nothing. Worse, the routing
step sometimes answered it with a clarifying question or a web search.

My memory layer adds a guard for it: it recognizes the question by meaning and answers it by date. The question is
compared with nine example phrasings ("what have we been talking about", "remind me what we discussed before", and so
on). At 0.82 or higher, the three most recent earlier conversations are shown, newest first, and the turn goes to plain
conversation whatever the routing step decided. In testing, paraphrases scored 0.82 to 1.0 (all but one of 13 cleared
0.82), specific questions like "what did we talk about regarding Barcelona" scored up to 0.815, and ordinary ones 0.77
or lower. The gap between "about the conversation" and "about a topic in it" is only 0.005, which is worth showing
rather than hiding. A misspelled "what are we discusing" scored 0.795, so a looser threshold of 0.76 also counts, but
only when the alternative was asking the user what they meant or refusing.

A smaller rule handles follow-ups. "What did he say?" names nothing searchable. When a message of six words or fewer
finds nothing on its own, recall tries again with the previous question attached. It only runs when the first search
found nothing, so an unrelated short question does not drag the last topic back in.

## What it costs

Most of the cost is structural, so it does not depend on the hardware:

- **Extra model calls added by memory:** 0 per turn.
- **Embedding calls:** 1 to 4 per chat turn (the question, the conversation check, the follow-up retry), each small and
  on two threads.
- **Tokens added when nothing relevant is found:** 0.
- **Tokens added when something is found:** at most about 200, plus the saved facts.
- **Work after the reply:** one background embedding of the new exchange note.
- **Work after a conversation ends:** one model call to write its summary.

For scale, on the laptop with the low-compute settings, one decision step of the 1.7B model took a median of about 3.1
seconds. Memory is not the main source of latency; the model calls are. The point of the design is that memory adds as
little as possible on top of them, and nothing at all on turns that do not need it. I have not timed the embedding and
search steps on their own.

## What is not solved

I would rather list these than have a reader find them.

- **The evaluation is small, and mine.** 12 self-written questions about the past over 19 real summaries, 16 ordinary
  questions, 14 detail questions. The thresholds were chosen by looking at the same data, so the results are optimistic.
  I have not run a public long-conversation benchmark such as LoCoMo or LongMemEval, or compared against an
  extraction-based memory library on the same hardware.
- **The thresholds belong to one embedding model.** Swap BGE-small and every number changes. The method carries over;
  the values do not.
- **Facts are never trimmed.** The cap keeps the list small, but 40 long facts could take more tokens than the whole chat
  budget, and only history is trimmed to fit.
- **Contradictions are handled only within facts, and only when the topic name matches.** Changing a favorite color
  does not touch old exchange notes or summaries that mention the old one.
- **A summary can delay a reply** if the user speaks while one is being written.
- **Private data flows into memory.** Mail and calendar results are kept out of logs, but a spoken reply that mentions
  them is saved, embedded and summarized. Raw turns are deleted after 30 days; summaries are kept.
- **Old details expire.** After 30 days only summaries remain.
- **Interrupted replies are not saved**, so an answer the user cut off leaves no note.
- **English only**, and **not separated per speaker**: the assistant can recognize who is talking, but memory is shared
  by the household.

## What I learned

1. **Budget prompt tokens and model calls, not storage.** Make the write path need no model call of its own and give the
   read path a hard size limit.
2. **Make abstaining the default.** A nearest neighbor always exists. Whether it deserves a place in the prompt is a
   separate question, and for a small model the answer is usually no.
3. **Measure the embedding model you actually use.** My starting threshold was badly wrong for my model. A few dozen
   hand-checked questions were enough to cut unnecessary memory from 15 of 16 ordinary turns to 4.

If you run small models on modest hardware, I would like to know whether these thresholds and rules hold on your data,
and especially where they break.

---

*Notes: all measurements were taken on a laptop CPU (Intel Core i5-13420H, no GPU) with low-compute settings, on my own
conversation data. This article was drafted with the help of Claude (Anthropic), working from my codebase and
measurements; the system, the design decisions and the measurements are my own.*
