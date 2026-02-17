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
        self.translated_confirmed = ""  # accumulated for debugging only
        self.translated_partial = ""
        self.context_pairs: list[tuple[str, str]] = []  # last 1 (source, translation) confirmed pair

    async def translate_confirmed(self, sentence: str, elapsed_ms: int = 0):
        word_count = len(sentence.split())
        context = self._build_context()
        translated = await self._call_gpt(sentence, f"CONFIRMED ({word_count}w, {len(self.context_pairs)}ctx)", context)
        if translated:
            self.translated_confirmed = (self.translated_confirmed + " " + translated).strip() if self.translated_confirmed else translated
            self.translated_partial = ""
            self.context_pairs.append((sentence, translated))
            if len(self.context_pairs) > 1:
                self.context_pairs.pop(0)
            if self.on_confirmed:
                await self.on_confirmed(translated, elapsed_ms)

    async def translate_partial(self, text: str, elapsed_ms: int = 0):
        word_count = len(text.split())
        context = self._build_context()
        translated = await self._call_gpt(text, f"PARTIAL ({word_count}w, {len(self.context_pairs)}ctx)", context)
        if translated:
            self.translated_partial = translated
            if self.on_partial:
                await self.on_partial(self.translated_partial, elapsed_ms)

    def _build_context(self) -> str:
        if not self.context_pairs:
            return ""
        lines = []
        for source, translation in self.context_pairs:
            lines.append(f"Source: {source}\nTranslation: {translation}")
        return "Previous context:\n" + "\n\n".join(lines)

    async def _call_gpt(self, text: str, label: str = "", context: str = "") -> str:
        try:
            t_start = time.monotonic()

            user_content = f"{context}\n\nTranslate: {text}" if context else text

            stream = await oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(lang=self.target_lang) + (
                        "\n\n" + self.tone_detector.get_tone_instruction() if self.tone_detector else ""
                    )},
                    {"role": "user", "content": user_content},
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
            ttft_str = f"{ttft:.0f}" if ttft is not None else "n/a"
            print(f"🌐 [{label}] ttft:{ttft_str}ms total:{total_ms:.0f}ms")
            print(f"    Source: {text}")
            print(f"    Result: {translated.strip()}")

            return translated.strip()

        except Exception as e:
            print(f"Translation error: {type(e).__name__}: {e}")
            return ""
