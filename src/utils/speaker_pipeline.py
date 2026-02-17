import re
import time
import asyncio
from utils.translation import Translator
from utils.translation_realtime import RealtimeTranslator
from utils.tone import ToneDetector
from utils.punctuation import SentenceSplitter

CONFIRM_PUNCT_COUNT = 1
PARTIAL_INTERVAL = 2
SILENCE_CONFIRM_SEC = 3.0


class SpeakerPipeline:
    """Per-speaker sentence confirmation and translation pipeline."""

    def __init__(self, speaker_id: str, on_confirmed, on_partial, on_confirmed_transcript, on_partial_transcript, target_lang: str, tone_detector: ToneDetector, stream_start: float = 0.0, confirm_punct_count: int = CONFIRM_PUNCT_COUNT, use_splitter: bool = True, partial_interval: int = PARTIAL_INTERVAL, use_realtime: bool = False):
        self.speaker_id = speaker_id
        self.confirmed_word_count = 0
        self.partial_count = 0
        self.prev_text = ""
        self.tone_detector = tone_detector
        self.confirm_punct_count = confirm_punct_count
        self.use_splitter = use_splitter
        self.partial_interval = partial_interval
        self.splitter = SentenceSplitter() if use_splitter else None
        self.on_confirmed_transcript = on_confirmed_transcript
        self.on_partial_transcript = on_partial_transcript
        self._silence_task: asyncio.Task | None = None
        self._stream_start: float = stream_start
        self._awaiting_new_partial: bool = True  # True at start and after each confirmation
        self._partial_start_ts: float = 0.0      # Wall-clock time when current partial sequence began
        self._last_partial_len: int = 0          # Word count of last partial sent for translation
        TranslatorClass = RealtimeTranslator if use_realtime else Translator
        self.translator = TranslatorClass(
            on_confirmed=on_confirmed,
            on_partial=on_partial,
            tone_detector=tone_detector,
            target_lang=target_lang,
        )

    def _elapsed_ms(self) -> int:
        """Milliseconds since stream started."""
        if not self._stream_start:
            return 0
        return int((time.time() - self._stream_start) * 1000)

    def _confirm_sentence(self, text: str, label: str):
        """Shared logic for confirming a sentence (punct, split, or silence)."""
        elapsed = self._elapsed_ms()
        now = time.time()
        latency = (now - self._partial_start_ts) * 1000 if self._partial_start_ts else 0
        print(f"⏱️  [{self.speaker_id}] CONFIRMED  ts={now:.3f}  elapsed={elapsed}ms  "
              f"partial_to_confirm={latency:.0f}ms  label={label}")
        print(f"✅ [{self.speaker_id}] {label}: \"{text}\" (elapsed={elapsed}ms)")
        self._awaiting_new_partial = True
        self._last_partial_len = 0
        loop = asyncio.get_event_loop()
        loop.create_task(self.translator.translate_confirmed(text, elapsed))
        if self.on_confirmed_transcript:
            loop.create_task(self.on_confirmed_transcript(text, elapsed))
        self.partial_count = 0

    def feed(self, full_text: str):
        """Feed the full accumulated text for this speaker. Runs confirmation + partial translation."""
        if full_text == self.prev_text:
            return
        self.prev_text = full_text

        self.tone_detector.feed_text(full_text)
        words = full_text.split()
        remaining_words = words[self.confirmed_word_count:]
        remaining_text = " ".join(remaining_words)
        print(f"🎤 [{self.speaker_id}] Remaining: \"{remaining_text}\"")

        # Sentence splitter: trigger async GPT if text is long and unpunctuated
        if self.splitter:
            self.splitter.check(remaining_words, self.confirmed_word_count)

        # Check for GPT-based split (direct cut, no punctuation needed)
        split_count = self.splitter.take_split(self.confirmed_word_count, remaining_words) if self.splitter else 0
        if split_count:
            new_confirmed = " ".join(remaining_words[:split_count])
            self.confirmed_word_count += split_count
            remaining_text = " ".join(words[self.confirmed_word_count:])
            self._confirm_sentence(new_confirmed, "confirmed (split)")

        # Check for confirmed sentence via natural punctuation
        matches = list(re.finditer(r'[.?!]\s+\w', remaining_text))
        if len(matches) >= self.confirm_punct_count:
            cut_match = matches[-self.confirm_punct_count]
            cut = cut_match.start() + 1
            new_confirmed = remaining_text[:cut].strip()

            if new_confirmed:
                self.confirmed_word_count += len(new_confirmed.split())
                remaining_text = " ".join(words[self.confirmed_word_count:])
                self._confirm_sentence(new_confirmed, "confirmed")

        # Send partial transcript every update for live display
        if remaining_text and self.on_partial_transcript:
            elapsed = self._elapsed_ms()
            now = time.time()
            if self._awaiting_new_partial:
                self._partial_start_ts = now
                self._awaiting_new_partial = False
                print(f"⏱️  [{self.speaker_id}] PARTIAL_START  ts={now:.3f}  elapsed={elapsed}ms")
            else:
                print(f"⏱️  [{self.speaker_id}] PARTIAL        ts={now:.3f}  elapsed={elapsed}ms")
            loop = asyncio.get_event_loop()
            loop.create_task(self.on_partial_transcript(remaining_text, elapsed))

        # Fire partial translation every N updates (skip ASR corrections — shorter than last partial)
        self.partial_count += 1
        remaining_word_count = len(remaining_text.split()) if remaining_text else 0
        if self.partial_count % self.partial_interval == 0 and remaining_text and remaining_word_count >= self._last_partial_len:
            self._last_partial_len = remaining_word_count
            elapsed = self._elapsed_ms()
            loop = asyncio.get_event_loop()
            loop.create_task(self.translator.translate_partial(remaining_text, elapsed))

        # Reset silence timer
        self._reset_silence_timer()

    def _reset_silence_timer(self):
        if self._silence_task:
            self._silence_task.cancel()
        self._silence_task = asyncio.get_event_loop().create_task(self._silence_confirm())

    async def _silence_confirm(self):
        await asyncio.sleep(SILENCE_CONFIRM_SEC)
        words = self.prev_text.split()
        remaining_words = words[self.confirmed_word_count:]
        remaining_text = " ".join(remaining_words)
        if not remaining_text:
            return
        self.confirmed_word_count += len(remaining_words)
        self._confirm_sentence(remaining_text, "silence auto-confirm")
