API for AST, ASR + translation

Update Log:

02 15 ~ 02 16

OpenAI Realtime API for translation:

- Implemented RealtimeTranslator using persistent WebSocket to OpenAI Realtime API
- All translations sent as out-of-band requests (`conversation: "none"`) with same 1ctx manual context
- Persistent WebSocket eliminates per-request connection/auth overhead
- Tested Realtime API models against Chat Completions baseline:

  | Approach                                           | TTFT avg | Max TTFT | Total avg | Notes                                                       |
  | -------------------------------------------------- | -------- | -------- | --------- | ----------------------------------------------------------- |
  | 1ctx chat completions (gpt-4o-mini)                | ~410ms   | 536ms    | ~600ms    | baseline                                                    |
  | 1ctx realtime (gpt-4o-mini-realtime-preview) run 1 | ~252ms   | 483ms    | ~307ms    | 25 samples                                                  |
  | 1ctx realtime (gpt-4o-mini-realtime-preview) run 2 | ~272ms   | 655ms    | —         | 40 samples                                                  |
  | 1ctx realtime (gpt-4o-mini-realtime-preview) run 3 | ~278ms   | 714ms    | ~360ms    | 50 samples, degrades over time (~194ms early → ~308ms late) |
  | 1ctx realtime (gpt-realtime) run 1                 | ~252ms   | 483ms    | ~335ms    |                                                             |
  | 1ctx realtime (gpt-realtime) run 2                 | ~355ms   | 1730ms   | —         | spiky, unstable                                             |
  | 1ctx realtime (gpt-realtime-mini) run 1            | ~215ms   | 467ms    | ~270ms    |                                                             |
  | 1ctx realtime (gpt-realtime-mini) run 2            | ~216ms   | 351ms    | ~283ms    | 27 samples, no spikes, no degradation                       |
  - All Realtime API models significantly faster than Chat Completions (~48-50% TTFT reduction)
  - `gpt-realtime-mini`: most consistent — ~216ms avg TTFT, max 351ms, stable throughout long sessions
  - `gpt-4o-mini-realtime-preview`: fast early but degrades over session, occasional spikes (687ms, 714ms)
  - `gpt-realtime`: inconsistent — stable on one run, spiky (1730ms) on another
  - Winner: `gpt-realtime-mini` — stable latency, no degradation, no spikes

- Also tested in-band conversation history (confirmed translations added to conversation):
  - CONFIRMED TTFT avg ~292ms vs ~259ms out-of-band — ~30ms slower due to server-side context processing
  - Overhead grows with conversation length (same issue as Responses API)
  - Translation quality comparable to manual 1ctx
  - Reverted to all out-of-band with manual 1ctx

- Added translator type toggle (Chat Completions / Realtime API) in frontend settings
- RealtimeTranslator uses `_send_lock` + `_creation_queue` for safe concurrent request tracking

Translation context window, configurable pipeline settings, per-speaker captions:

- Added previous confirmed source+translation pair as context to GPT translation prompt for better coherence
- Tested 0ctx, 1ctx, and 3ctx and chose 1ctx based on latency/quality tradeoff:

  | Approach                             | TTFT avg | Max TTFT | Total avg |
  | ------------------------------------ | -------- | -------- | --------- |
  | 0ctx (chat completions)              | ~330ms   | 528ms    | ~560ms    |
  | 1ctx single msg (chat completions)   | ~410ms   | 536ms    | ~600ms    |
  | 1ctx multi-turn (chat completions)   | ~540ms   | 946ms    | ~730ms    |
  | 3ctx single msg (chat completions)   | ~570ms   | 1843ms   | ~830ms    |
  | Responses API (previous_response_id) | ~790ms   | 1460ms   | ~1050ms   |
  - 3ctx: dangerous spikes (1.8s TTFT), ~70% overhead
  - Multi-turn: message structure overhead worse than single-message context
  - Responses API: server-side state management adds ~380ms overhead, not worth it for 1 context pair
  - Winner: 1ctx single msg — only ~80ms overhead, no spikes, good translation consistency (pronouns, terminology, style)
  - Context formatted as "Previous context:\nSource: ...\nTranslation: ...\n\nTranslate: ..." in user message

- Added configurable aggressiveness setting (passed as number via query param):
  - 1 (high): confirm_punct_count=1, sentence splitter ON — faster confirmations
  - 2 (low): confirm_punct_count=2, sentence splitter OFF — waits for more punctuation
- Added configurable update frequency (partial_interval) via query param
- Added configurable media delay (video + audio + caption scheduling) from frontend
- Per-speaker caption boxes in content script with distinct colors per speaker (green, blue, pink, yellow, purple)
- Translation caption: fixed-height scrollable box, confirmed in speaker color, partial on new line, "..." placeholder between confirmed and next partial to prevent layout jump

02 14

Tested Deepgram Nova stream diarization, Assembly AI API

02 13

GPT base sentence splitter for increasing sentences w/o punctuation, live transcript streaming, silence auto-confirm:

- Added SentenceSplitter (gpt-4.1-nano, Responses API, structured JSON output) to split long unpunctuated STT streams at semantic boundaries
- Added confirmed_transcript and partial_transcript WebSocket messages for live source text display alongside translations
- Partial transcript fires every feed() call for real-time display; partial translation stays throttled (every 2nd call)
- Auto-confirm remaining unconfirmed text after 3 seconds of speaker silence — prevents dangling text when speaker stops mid-sentence
- Splitter triggers when unconfirmed text exceeds 15 words with no punctuation; staleness guard discards result if natural confirmation happened during GPT latency
- Wired on_confirmed_transcript and on_partial_transcript callbacks in both speechmatics and elevenlabs routers

02 12

Speaker diarization and incremental translation:

- Speechmatics router now uses SpeakerPipeline for per-speaker sentence confirmation and translation
- ElevenLabs router refactored to use SpeakerPipeline (single "default" speaker, no native diarization)
- confirmed_translation now sends only the new incremental translation, not the full accumulated text
- Translator.translated_confirmed kept for debugging only, no longer sent to frontend
- All WebSocket messages to frontend now include `speaker` field for speaker identification
- Added full Speechmatics source language support (56 languages including bilingual/multilingual)

02 11

Configurable source and target language:

- Target language (translation output) is now configurable from the frontend via query param `target_lang`
- Source language (ASR input) is configurable via query param `source_lang` for Speechmatics; ElevenLabs auto-detects
- Translator accepts `target_lang` parameter
- ToneDetector is now language-aware: Korean and Japanese get language-specific tone instructions, others get generic tone guidance
- Renamed all Korean/English-specific variable names to language-neutral equivalents (`confirmed_korean` → `confirmed_translation`, `english_source` → `source_text`, etc.)

Multi-provider STT support:

- Added ElevenLabs Scribe v2 as an alternative STT provider alongside Speechmatics
- Both providers share the same translation pipeline (translationExp.py + tone.py)
- Client connects to /stt/speechmatics or /stt/elevenlabs to choose provider
- ElevenLabs router (elevenlabs.py) uses raw WebSocket proxy to ElevenLabs API
- Speechmatics router (speechmatics.py) uses speechmatics-rt SDK with event callbacks
- Same sentence boundary detection, confirmed/partial translation, and tone detection

02 10

Architecture - Sentence-based translation pipeline:

Sentence Segmentation (speechmatics.py)

- Punctuation-based sentence confirmation from STT stream
- Tracks confirmed_word_count pointer into full_text, rebuilds remaining_text each update
- Configurable CONFIRM_PUNCT_COUNT (default 2): number of punctuation+text matches needed
- Confirmed sentences sent to translator once — permanent, never retranslated
- Partial (remaining) text translated every PARTIAL_INTERVAL (default 6) STT updates as preview
- Eliminates overlap re-translation problem entirely: no combiner, no fuzzy matching

Translation (translationExp.py)

- Two async translation paths: translate_confirmed() and translate_partial()
- partial_stale boolean: set True on confirmed, prevents stale partial results from displaying
- translated_confirmed grows permanently, translated_partial is temporary preview
- Websocket callbacks send confirmed_translation and partial_translation to frontend

Why this approach:

- Previous rolling-window + combiner failed because GPT re-translates overlapping
  English words differently each time, making fuzzy Korean stitching unreliable
- Boundary locking failed because STT glitches get permanently locked and GPT
  misaligns markers with small context segments
- Sentence-based approach: each GPT call gets a complete thought with full context,
  no overlap, no stitching, no accumulation bugs

02 05 ~ 02 09

Performance - Architectural latency optimization:

Latency Architecture

- Total pipeline: ~600-900ms (translator only, combiner instant)
- Latency does NOT stack: each translation is independent, not queued
- 3-partial interval (~1.5s) > translation time (~0.9s) = no queue buildup
- Combiner searches only last 40 words of combined (capped, O(1) vs text length)
- combined_text grows unbounded but only tail is ever searched
- Stale translation check (seq < latest_seq) drops outdated results
- Tested every-2-partial cadence; reverted to every-3 — marginal UX gain
  did not justify 50% more API calls and less stable anchor matching
- Early 1st-partial trigger cuts initial display latency from ~1.5s to ~0.7s
- Tone detection runs concurrently as fire-and-forget, zero blocking

New implementation - tone detection, caption UX:

Tone Detection (tone.py)

- LLM-based Korean speech register detection (casual/casual_polite/formal/narrative - ~해... /~해요... /~합니다... /~한다...)
- Runs once after 30 words accumulated, concurrent via asyncio.create_task
- Zero pipeline latency impact; dynamically updates translator system prompt
- Correctly maps stream types (Twitch → casual, YouTube → casual_polite, etc.)

Translator (translationTest.py)

- Early trigger: translate on 1st partial for fast initial display (~0.7s)
- Normal cadence: every 3 partials after that (~1.5s between updates)
- Dynamic prompt rebuilds with tone instruction when tone changes

Transcript Accumulation (elevenlabsText.py)

- Accumulate all committed transcripts into running history
- Feed committed_text + current_partial to translator
- Fixes duplication bug where ElevenLabs resets partial window after silence
- Rolling window now slides over continuous text with no gaps

Combiner (combiner.py)

- Short combined (≤5 words) → replace instead of merge (fixes early duplication)
- Minimum anchor length: 3 words (prevents false positives on common phrases)
- Search depth: 40 words (was 20)
- Weighted scoring: similarity × (prefix_len / max_prefix) prefers longer matches

Caption Overlay (content.ts)

- Scrolling teleprompter: 2-line max visible window with overflow:hidden
- Typewriter effect on new characters (25ms/char)
- Character-level diffing: stable prefix stays untouched (no flicker)
- Smooth translateY scroll as new text pushes old text up
- Overflow protection: >50 new chars → show bulk instantly, typewrite last 30

02 04
Performance Optimizations: skip processing for minor/duplicate ASR updates

- Track last processed transcript to detect redundant updates
- Skip identical transcripts (common during pauses)
- Skip when only 1-2 characters added (still typing same word)
- Always reprocess when transcript gets shorter (ASR correction)
- Eliminates ~50-70% of redundant embedding/GPT calls during stable speech

02 03

Performance Optimizations:

- Punctuation-only boundaries: only encode prefixes ending with sentence
  punctuation (.!?) instead of all word prefixes (reduced from N prefixes
  to 2-4 sentence boundaries, ~5-7x fewer computations)
- Batch encoding: encode all prefixes + Korean in single model.encode() call
  (reduced from N+1 calls to 1 call per ASR update)
- Add debug flag (default False) to disable verbose logging in production
- Wrap 100+ print statements in debug checks to eliminate I/O blocking

01 27 ~ 02 03

Implement cross-language embedding matching to avoid re-translating
confirmed content, significantly reducing latency and API costs.

Core Implementation:

- Add LaBSE (Language-agnostic BERT Sentence Embeddings) for comparing
  English source text against confirmed Korean translations
- Use prefix-based similarity scoring: generate all prefixes of English
  transcript and find which prefix best matches the confirmed Korean
- Preload model at server startup via lifespan context manager

Boundary Detection Algorithm:

- Calculate similarity scores for all English prefixes against Korean
- Find peak similarity score among all prefixes
- Among prefixes ending with sentence punctuation (.!?), select the one
  closest to peak score (not just the longest match)
- This prevents boundary from extending past sentence endings

Confirmation Logic (2-cycle system):

- Sentences require 2 consecutive ASR updates with punctuation to confirm
- Prevents premature confirmation of incomplete/unstable translations
- Confirmed sentences accumulate in confirmed_korean for embedding matching

GPT Prompt Fix for Incomplete Sentences:

- GPT was adding punctuation to incomplete English sentences, causing
  false confirmations (e.g., "but it won't be on the" → "~에 있지 않을 것입니다.")
- Added prompt instruction: preserve incompleteness if source lacks
  ending punctuation, include punctuation if source has it
- Prevents incomplete translations from triggering confirmation logic

Bug Fixes:

- Fix model double-loading due to inconsistent import paths (relative vs
  absolute imports between modules)
- Fix embedding boundary extending past sentence punctuation by selecting
  punctuated prefix closest to peak rather than walking back with tolerance
- Fix confirmed_korean being overwritten instead of appended when
  translating remaining English portion

Files:

- src/utils/embeddings.py: LaBSE loading, batch prefix matching
- src/utils/translationTwo.py: streaming translation with confirmation logic
- main.py: model preloading in lifespan
