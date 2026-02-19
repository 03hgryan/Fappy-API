import os
import time
import asyncio
from openai import AsyncOpenAI

oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TARGET_LANG = "Korean"

SYSTEM_PROMPT = """You are a real-time subtitle translator for live audio. Translate to {lang}.

The source text is auto-generated speech recognition, which may contain errors, mishearings, or awkward phrasing. Your job is to convey what the speaker *meant*, not to literally translate the raw transcript.

Rules:
- Translate the speaker's intent, not the literal text
- If the transcript looks garbled or nonsensical, infer the likely meaning from context and translate that
- Produce natural, fluent output as if a native {lang} speaker were explaining the same idea
- Match the speaker's tone and energy
- Output ONLY the translation, nothing else"""

SUMMARY_PROMPT = """Summarize the following transcript in under 30 words. Focus on subject matter, key terms, entities, and the speaker's current point.

Transcript:
{transcript}

Summary:"""


class Translator:
    def __init__(self, on_confirmed=None, on_partial=None, on_partial_delta=None, tone_detector=None, target_lang=None):
        self.on_confirmed = on_confirmed
        self.on_partial = on_partial
        self.on_partial_delta = on_partial_delta
        self.tone_detector = tone_detector
        self.target_lang = target_lang or TARGET_LANG
        self.translated_confirmed = ""  # accumulated for debugging only
        self.translated_partial = ""
        self.context_pairs: list[tuple[str, str]] = []  # last 1 (source, translation) confirmed pair
        self.confirmed_transcript = ""  # full accumulated transcript for summary
        self.topic_summary = ""
        self._summary_task: asyncio.Task | None = None
        self._partial_generation: int = 0

    async def translate_confirmed(self, sentence: str, elapsed_ms: int = 0):
        word_count = len(sentence.split())
        context = self._build_context()
        translated = await self._call_gpt(sentence, f"CONFIRMED ({word_count}w)", context)
        if translated:
            self.translated_confirmed = (self.translated_confirmed + " " + translated).strip() if self.translated_confirmed else translated
            self.translated_partial = ""
            self.context_pairs.append((sentence, translated))
            if len(self.context_pairs) > 1:
                self.context_pairs.pop(0)
            self.confirmed_transcript = (self.confirmed_transcript + " " + sentence).strip() if self.confirmed_transcript else sentence
            self._update_summary_async()
            if self.on_confirmed:
                await self.on_confirmed(translated, elapsed_ms)

    async def translate_partial(self, text: str, elapsed_ms: int = 0):
        self._partial_generation += 1
        gen = self._partial_generation
        word_count = len(text.split())
        context = self._build_context()
        translated = await self._call_gpt(
            text, f"PARTIAL ({word_count}w)", context,
            delta_cb=self.on_partial_delta, generation=gen, elapsed_ms=elapsed_ms,
        )
        if translated and gen == self._partial_generation:
            self.translated_partial = translated
            if self.on_partial:
                await self.on_partial(self.translated_partial, elapsed_ms)

    def _build_context(self) -> str:
        parts = []
        if self.topic_summary:
            parts.append(f"Topic: {self.topic_summary}")
        if self.context_pairs:
            source, translation = self.context_pairs[-1]
            parts.append(f"Previous source: {source}\nPrevious translation: {translation}")
        if not parts:
            return ""
        return "\n".join(parts)

    def _update_summary_async(self):
        if self._summary_task and not self._summary_task.done():
            self._summary_task.cancel()
        self._summary_task = asyncio.create_task(self._generate_summary())

    async def _generate_summary(self):
        try:
            result = await oai.responses.create(
                model="gpt-4.1-nano",
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

    async def _call_gpt(self, text: str, label: str = "", context: str = "",
                        delta_cb=None, generation: int = 0, elapsed_ms: int = 0) -> str:
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
                    if delta_cb and generation == self._partial_generation:
                        await delta_cb(delta, generation, elapsed_ms)

            total_ms = (time.monotonic() - t_start) * 1000
            ttft_str = f"{ttft:.0f}" if ttft is not None else "n/a"
            print(f"🌐 [{label}] ttft:{ttft_str}ms total:{total_ms:.0f}ms")
            print(f"    Source: {text}")
            print(f"    Result: {translated.strip()}")

            return translated.strip()

        except Exception as e:
            print(f"Translation error: {type(e).__name__}: {e}")
            return ""
