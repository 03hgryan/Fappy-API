API for AST, ASR + translation

Update Log:

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
