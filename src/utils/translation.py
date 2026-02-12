import os
import time
from openai import AsyncOpenAI

oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TARGET_LANG = "Korean"

SYSTEM_PROMPT = """You are a real-time subtitle translator. Translate the given text to {lang}.

Rules:
- Translate naturally, not word-by-word
- Match the speaker's energy and intent
- Output ONLY the translation, nothing else
- No quotes, no explanations, no labels"""


class Translator:
    def __init__(self, on_confirmed=None, on_partial=None, tone_detector=None, target_lang=None):
        self.on_confirmed = on_confirmed
        self.on_partial = on_partial
        self.tone_detector = tone_detector
        self.target_lang = target_lang or TARGET_LANG
        self.translated_confirmed = ""  # not used: accumulated for debugging only
        self.translated_partial = ""
        self.partial_stale = False

    async def translate_confirmed(self, sentence: str):
        self.partial_stale = True
        translated = await self._call_gpt(sentence)
        if translated:
            self.translated_confirmed = (self.translated_confirmed + " " + translated).strip() if self.translated_confirmed else translated
            self.translated_partial = ""
            print(f"✅🌐 Source: {sentence}")
            print(f"    Translated: {translated}")
            if self.on_confirmed:
                await self.on_confirmed(translated)

    async def translate_partial(self, text: str):
        self.partial_stale = False
        translated = await self._call_gpt(text)
        if translated and not self.partial_stale:
            self.translated_partial = translated
            print(f"⏳🌐 Source: {text}")
            print(f"    Translated: {translated}")
            if self.on_partial:
                await self.on_partial(self.translated_partial)

    async def _call_gpt(self, text: str) -> str:
        try:
            t_start = time.monotonic()

            stream = await oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(lang=self.target_lang) + (
                        "\n\n" + self.tone_detector.get_tone_instruction() if self.tone_detector else ""
                    )},
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
                max_tokens=200,
                stream=True,
            )

            translated = ""
            ttft = None
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    if ttft is None:
                        ttft = (time.monotonic() - t_start) * 1000
                    translated += delta

            total_ms = (time.monotonic() - t_start) * 1000
            print(f"    ⏱️ ttft:{ttft:.0f}ms total:{total_ms:.0f}ms")

            return translated.strip()

        except Exception as e:
            print(f"Translation error: {type(e).__name__}: {e}")
            return ""