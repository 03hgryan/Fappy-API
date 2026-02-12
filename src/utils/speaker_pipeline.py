import re
import asyncio
from utils.translation import Translator
from utils.tone import ToneDetector

CONFIRM_PUNCT_COUNT = 1
PARTIAL_INTERVAL = 2


class SpeakerPipeline:
    """Per-speaker sentence confirmation and translation pipeline."""

    def __init__(self, speaker_id: str, on_confirmed, on_partial, target_lang: str, tone_detector: ToneDetector):
        self.speaker_id = speaker_id
        self.confirmed_word_count = 0
        self.partial_count = 0
        self.prev_text = ""
        self.tone_detector = tone_detector
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
        remaining_text = " ".join(words[self.confirmed_word_count:])

        # Check for confirmed sentence
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
                self.partial_count = 0

        # Fire partial translation every N updates
        self.partial_count += 1
        if self.partial_count % PARTIAL_INTERVAL == 0 and remaining_text:
            loop = asyncio.get_event_loop()
            loop.create_task(self.translator.translate_partial(remaining_text))
