# Pi review findings (2026-10-24)

Notes from a read-through of `main` @ `3625517`. Nothing here is measured on the Pi; every item is a lead to check
there first. Test baseline on this commit: 798 passed, 5 failed (`test_tts_selection`, only because Piper/pyttsx3 are
not installed), 10 skipped.

## Inference

1. **Decode uses only 2 of 4 cores.** `config/inference.yaml` has `n_threads: null`, so llama-cpp-python falls back to
   its default of half the cores (`cpu_count() // 2`) for generation. On the Pi 5 that is 2. Set `n_threads: 4` and
   measure the control-decode median before/after.
2. **`num_batch` is ignored for llama.cpp.** `LlamaCppClient` never passes it to `llama_cpp.Llama(...)`
   (`src/tpa/inference/llama_cpp_client.py`). It currently equals the default (512), so nothing is broken, but changing
   it has no effect.
3. **Prompt-prefix cache may be too small.** `prompt_cache_mb: 768` assumes ~70 MB per cached prefix. Estimate for the
   current ~2,600-token control prefix with f16 KV: ~300 MB (Qwen3-1.7B: 28 layers x 8 KV heads x 128 dim) to ~375 MB
   (Qwen3-4B: 36 layers). If right, 768 MB holds only ~2 prefixes while control, chat and narrate prompts alternate, and
   they may evict each other. Log cache hits on the device; consider a larger cache or a quantized KV cache.

## Voice loop

4. **Fixed acknowledgements add delay.** "Hey there, what's up?" is spoken in full before the command window opens
   (`voice_session.py`). "Working on that." fires after 1 s (`_PROCESSING_ACK_DELAY_SEC`); on the Pi a control decode
   likely takes longer than that on nearly every turn.
5. **Everything shares 4 cores.** Wake word (openwakeword, every 80 ms), Whisper (plus its own VAD filter and the
   spectral noise suppressor), embeddings and the LLM compete. Embeddings are capped at 2 ONNX threads; Whisper and
   openwakeword are not capped.

## Architecture / open items carried from the stage docs

6. Router and chat share one model path; running the tuned 1.7B router beside an untouched chat model needs a code change.
7. The tuned router only works with the exact prompt it was trained on (`control_stage.md` + tool text).

## Stale docs

8. `docs/HANDOFF.md` says `main` is the old routing design (it now has the unified orchestrator), gives the
   `ramgopalraju9/vedic-atlas` clone URL (repo is now `praveensivaram/vedic-atlas`), and quotes 392 tests (now 798).
