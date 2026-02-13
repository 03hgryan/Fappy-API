import re
import asyncio
from utils.translation import Translator
from utils.tone import ToneDetector
from utils.punctuation import SentenceSplitter

CONFIRM_PUNCT_COUNT = 1
PARTIAL_INTERVAL = 2
SILENCE_CONFIRM_SEC = 3.0


class SpeakerPipeline:
    """Per-speaker sentence confirmation and translation pipeline."""

    def __init__(self, speaker_id: str, on_confirmed, on_partial, on_confirmed_transcript, on_partial_transcript, target_lang: str, tone_detector: ToneDetector):
        self.speaker_id = speaker_id
        self.confirmed_word_count = 0
        self.partial_count = 0
        self.prev_text = ""
        self.tone_detector = tone_detector
        self.splitter = SentenceSplitter()
        self.on_confirmed_transcript = on_confirmed_transcript
        self.on_partial_transcript = on_partial_transcript
        self._silence_task: asyncio.Task | None = None
        self.translator = Translator(
            on_confirmed=on_confirmed,
            on_partial=on_partial,
            tone_detector=tone_detector,
            target_lang=target_lang,
        )

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
        self.splitter.check(remaining_words, self.confirmed_word_count)

        # Check for GPT-based split (direct cut, no punctuation needed)
        split_count = self.splitter.take_split(self.confirmed_word_count, remaining_words)
        if split_count:
            new_confirmed = " ".join(remaining_words[:split_count])
            self.confirmed_word_count += split_count
            remaining_text = " ".join(words[self.confirmed_word_count:])
            print(f"✅🔤 [{self.speaker_id}] confirmed (split): \"{new_confirmed}\"")
            loop = asyncio.get_event_loop()
            loop.create_task(self.translator.translate_confirmed(new_confirmed))
            if self.on_confirmed_transcript:
                loop.create_task(self.on_confirmed_transcript(new_confirmed))
            self.partial_count = 0

        # Check for confirmed sentence via natural punctuation
        matches = list(re.finditer(r'[.?!]\s+\w', remaining_text))
        if len(matches) >= CONFIRM_PUNCT_COUNT:
            cut_match = matches[-CONFIRM_PUNCT_COUNT]
            cut = cut_match.start() + 1
            new_confirmed = remaining_text[:cut].strip()

            if new_confirmed:
                self.confirmed_word_count += len(new_confirmed.split())
                remaining_text = " ".join(words[self.confirmed_word_count:])
                print(f"✅ [{self.speaker_id}] confirmed: \"{new_confirmed}\"")
                loop = asyncio.get_event_loop()
                loop.create_task(self.translator.translate_confirmed(new_confirmed))
                if self.on_confirmed_transcript:
                    loop.create_task(self.on_confirmed_transcript(new_confirmed))
                self.partial_count = 0

        # Send partial transcript every update for live display
        if remaining_text and self.on_partial_transcript:
            loop = asyncio.get_event_loop()
            loop.create_task(self.on_partial_transcript(remaining_text))

        # Fire partial translation every N updates
        self.partial_count += 1
        if self.partial_count % PARTIAL_INTERVAL == 0 and remaining_text:
            loop = asyncio.get_event_loop()
            loop.create_task(self.translator.translate_partial(remaining_text))

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
        print(f"⏱️ [{self.speaker_id}] silence auto-confirm: \"{remaining_text}\"")
        self.confirmed_word_count += len(remaining_words)
        loop = asyncio.get_event_loop()
        loop.create_task(self.translator.translate_confirmed(remaining_text))
        if self.on_confirmed_transcript:
            loop.create_task(self.on_confirmed_transcript(remaining_text))
        self.partial_count = 0
