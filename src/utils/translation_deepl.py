import os
import time
import asyncio
import httpx
from openai import AsyncOpenAI
from utils.tone import TONE_INSTRUCTIONS_KOREAN, TONE_INSTRUCTIONS_JAPANESE, TONE_INSTRUCTIONS_GENERIC, DEFAULT_TONE
from utils.languages import TARGET_LANG_MAP, FORMALITY_SUPPORTED_LANGS, CUSTOM_INSTRUCTION_LANGS
from utils.translation_realtime import RealtimeTranslator

_oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "")
DEEPL_BASE_URL = os.getenv("DEEPL_BASE_URL", "https://api-free.deepl.com")

# Map tone detector results to DeepL formality
TONE_TO_FORMALITY = {
    "casual": "prefer_less",
    "casual_polite": "default",
    "formal": "prefer_more",
    "narrative": "default",
}

CUSTOM_INSTRUCTIONS = [
    "The source text is auto-generated speech recognition which may contain errors or mishearings.",
    "Translate the speaker's intent, not the literal text. Infer meaning from context if the transcript is garbled.",
    "Produce natural, fluent output as if a native speaker were explaining the same idea.",
]

SUMMARY_PROMPT = """Summarize the following transcript in under 30 words. Focus on subject matter, key terms, entities, and the speaker's current point.

Transcript:
{transcript}

Summary:"""


class DeepLTranslator:
    """Translator using DeepL text translation API with free context injection."""

    def __init__(self, on_confirmed=None, on_partial=None, on_partial_delta=None, tone_detector=None, target_lang=None):
        self.on_confirmed = on_confirmed
        self.on_partial = on_partial
        self.on_partial_delta = on_partial_delta
        self.tone_detector = tone_detector
        self.target_lang = target_lang or "Korean"
        self.deepl_target = TARGET_LANG_MAP.get(self.target_lang, "EN-US")
        self.translated_confirmed = ""
        self.translated_partial = ""
        self.context_pairs: list[tuple[str, str]] = []
        self.confirmed_transcript = ""
        self.topic_summary = ""
        self._summary_task: asyncio.Task | None = None
        self._client = httpx.AsyncClient(timeout=10.0, http2=True)
        # Use GPT realtime for partial translations (cheaper than DeepL)
        self._rt = RealtimeTranslator(
            on_confirmed=None,
            on_partial=on_partial,
            on_partial_delta=on_partial_delta,
            tone_detector=tone_detector,
            target_lang=target_lang,
        )

    async def translate_confirmed(self, sentence: str, elapsed_ms: int = 0):
        word_count = len(sentence.split())
        context = self._build_context()
        translated = await self._call_deepl(
            sentence, f"CONFIRMED ({word_count}w)", context,
            model_type="quality_optimized",
        )
        if translated:
            self.translated_confirmed = (
                (self.translated_confirmed + " " + translated).strip()
                if self.translated_confirmed else translated
            )
            self.translated_partial = ""
            self.context_pairs.append((sentence, translated))
            if len(self.context_pairs) > 1:
                self.context_pairs.pop(0)
            self.confirmed_transcript = (
                (self.confirmed_transcript + " " + sentence).strip()
                if self.confirmed_transcript else sentence
            )
            self._update_summary_async()
            if self.on_confirmed:
                await self.on_confirmed(translated, elapsed_ms)

    async def translate_partial(self, text: str, elapsed_ms: int = 0):
        # Sync context so the realtime translator has the latest DeepL-confirmed pairs
        self._rt.context_pairs = self.context_pairs
        self._rt.topic_summary = self.topic_summary
        await self._rt.translate_partial(text, elapsed_ms)
        self.translated_partial = self._rt.translated_partial

    def _build_context(self) -> str:
        parts = []
        if self.topic_summary:
            parts.append(f"Topic: {self.topic_summary}")
        if self.context_pairs:
            source, translation = self.context_pairs[-1]
            parts.append(f"Previous source: {source}\nPrevious translation: {translation}")
        if not parts:
            return ""
        return "\n\n".join(parts)

    def _get_formality(self) -> str:
        if self.tone_detector:
            tone = self.tone_detector.current_tone
            return TONE_TO_FORMALITY.get(tone, "default")
        return "default"

    def _get_tone_instruction(self) -> str:
        """Get tone instruction based on detected tone and target language."""
        tone = self.tone_detector.current_tone if self.tone_detector else DEFAULT_TONE
        if self.target_lang == "Korean":
            return TONE_INSTRUCTIONS_KOREAN.get(tone, "")
        elif self.target_lang == "Japanese":
            return TONE_INSTRUCTIONS_JAPANESE.get(tone, "")
        return TONE_INSTRUCTIONS_GENERIC.get(tone, "")

    async def _call_deepl(self, text: str, label: str = "", context: str = "", model_type: str = "quality_optimized") -> str:
        try:
            t_start = time.monotonic()

            body: dict = {
                "text": [text],
                "target_lang": self.deepl_target,
                "split_sentences": "0",
                "model_type": model_type,
            }

            if context:
                body["context"] = context

            formality = self._get_formality()
            if formality != "default" and self.deepl_target in FORMALITY_SUPPORTED_LANGS:
                body["formality"] = formality

            # custom_instructions only supported for certain target languages
            # and only with quality_optimized model
            if self.deepl_target.split("-")[0] in CUSTOM_INSTRUCTION_LANGS and model_type == "quality_optimized":
                instructions = list(CUSTOM_INSTRUCTIONS)
                # Add tone instruction for Korean/Japanese instead of formality param
                tone_instruction = self._get_tone_instruction()
                if tone_instruction:
                    instructions.append(tone_instruction)
                body["custom_instructions"] = instructions

            resp = await self._client.post(
                f"{DEEPL_BASE_URL}/v2/translate",
                json=body,
                headers={
                    "Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()

            data = resp.json()
            translated = data["translations"][0]["text"]

            total_ms = (time.monotonic() - t_start) * 1000
            print(f"🌐 [{label}] total:{total_ms:.0f}ms (deepl/{model_type})")
            print(f"    Source: {text}")
            print(f"    Result: {translated}")

            return translated

        except Exception as e:
            print(f"DeepL error: {type(e).__name__}: {e}")
            return ""

    def _update_summary_async(self):
        if self._summary_task and not self._summary_task.done():
            self._summary_task.cancel()
        self._summary_task = asyncio.create_task(self._generate_summary())

    async def _generate_summary(self):
        try:
            result = await _oai.responses.create(
                model="gpt-4o-mini",
                input=SUMMARY_PROMPT.format(transcript=self.confirmed_transcript),
                temperature=0,
                max_output_tokens=60,
            )
            new_summary = result.output_text.strip()
            if new_summary:
                self.topic_summary = new_summary
                print(f"📝 Summary: {new_summary}")
        except Exception as e:
            print(f"Summary error: {type(e).__name__}: {e}")

    async def close(self):
        await self._rt.close()
        await self._client.aclose()
