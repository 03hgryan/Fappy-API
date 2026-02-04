API for AST, ASR + translation

Update Log:

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
