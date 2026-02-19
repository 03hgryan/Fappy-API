import os
import json
import time
import asyncio
import websockets
from openai import AsyncOpenAI

_oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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


class _ResponseTracker:
    __slots__ = ("label", "source", "future", "text", "ttft", "start_time",
                 "delta_cb", "generation", "elapsed_ms", "translator")

    def __init__(self, label: str, source: str, delta_cb=None,
                 generation: int = 0, elapsed_ms: int = 0, translator=None):
        self.label = label
        self.source = source
        self.future: asyncio.Future = asyncio.get_event_loop().create_future()
        self.text = ""
        self.ttft: float | None = None
        self.start_time = time.monotonic()
        self.delta_cb = delta_cb
        self.generation = generation
        self.elapsed_ms = elapsed_ms
        self.translator = translator


class RealtimeTranslator:
    """Translator using OpenAI Realtime API over persistent WebSocket."""

    def __init__(self, on_confirmed=None, on_partial=None, on_partial_delta=None, tone_detector=None, target_lang=None):
        self.on_confirmed = on_confirmed
        self.on_partial = on_partial
        self.on_partial_delta = on_partial_delta
        self.tone_detector = tone_detector
        self.target_lang = target_lang or "Korean"
        self.translated_confirmed = ""
        self.translated_partial = ""
        self.context_pairs: list[tuple[str, str]] = []
        self.confirmed_transcript = ""  # full accumulated transcript for summary
        self.topic_summary = ""
        self._summary_task: asyncio.Task | None = None
        self._partial_generation: int = 0

        self._ws = None
        self._connected = False
        self._connect_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._reader_task: asyncio.Task | None = None
        self._response_map: dict[str, _ResponseTracker] = {}
        self._creation_queue: asyncio.Queue[_ResponseTracker] = asyncio.Queue()

    async def _ensure_connected(self):
        if self._connected:
            return
        async with self._connect_lock:
            if self._connected:
                return
            await self._connect()

    async def _connect(self):
        api_key = os.getenv("OPENAI_API_KEY")
        url = "wss://api.openai.com/v1/realtime?model=gpt-realtime-mini"
        # url = "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview-2024-12-17"
        self._ws = await websockets.connect(
            url,
            additional_headers={
                "Authorization": f"Bearer {api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
        )
        # Configure session: text-only
        await self._ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "temperature": 0.6,
                "max_response_output_tokens": 200,
            }
        }))
        self._reader_task = asyncio.create_task(self._read_loop())
        self._connected = True
        print("🔌 Realtime API WebSocket connected")

    def _build_instructions(self) -> str:
        prompt = SYSTEM_PROMPT.format(lang=self.target_lang)
        if self.tone_detector:
            tone = self.tone_detector.get_tone_instruction()
            if tone:
                prompt += "\n\n" + tone
        return prompt

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

    async def _read_loop(self):
        try:
            async for raw in self._ws:
                event = json.loads(raw)
                t = event.get("type", "")

                if t == "response.created":
                    resp_id = event["response"]["id"]
                    try:
                        tracker = self._creation_queue.get_nowait()
                        self._response_map[resp_id] = tracker
                    except asyncio.QueueEmpty:
                        pass

                elif t == "response.text.delta":
                    resp_id = event.get("response_id")
                    tracker = self._response_map.get(resp_id)
                    if tracker:
                        if tracker.ttft is None:
                            tracker.ttft = (time.monotonic() - tracker.start_time) * 1000
                        delta_text = event.get("delta", "")
                        tracker.text += delta_text
                        if (delta_text and tracker.delta_cb and tracker.translator
                                and tracker.generation == tracker.translator._partial_generation):
                            asyncio.create_task(tracker.delta_cb(
                                delta_text, tracker.generation, tracker.elapsed_ms))

                elif t == "response.text.done":
                    resp_id = event.get("response_id")
                    tracker = self._response_map.get(resp_id)
                    if tracker:
                        tracker.text = event.get("text", tracker.text)

                elif t == "response.done":
                    resp_id = event.get("response", {}).get("id")
                    tracker = self._response_map.pop(resp_id, None)
                    if tracker and not tracker.future.done():
                        total = (time.monotonic() - tracker.start_time) * 1000
                        ttft_str = f"{tracker.ttft:.0f}" if tracker.ttft else "n/a"
                        print(f"🌐 [{tracker.label}] ttft:{ttft_str}ms total:{total:.0f}ms")
                        print(f"    Source: {tracker.source}")
                        print(f"    Result: {tracker.text.strip()}")
                        tracker.future.set_result(tracker.text.strip())

                elif t == "error":
                    error = event.get("error", {})
                    print(f"❌ Realtime API error: {error.get('message', error)}")

                elif t == "session.created":
                    print("✅ Realtime session created")

                elif t == "session.updated":
                    print("✅ Realtime session updated")

        except websockets.ConnectionClosed as e:
            print(f"❌ Realtime WebSocket closed: {e}")
        except Exception as e:
            print(f"❌ Realtime reader error: {type(e).__name__}: {e}")
        finally:
            self._connected = False
            for tracker in self._response_map.values():
                if not tracker.future.done():
                    tracker.future.set_result("")
            self._response_map.clear()

    async def _send_translation(self, text: str, label: str,
                                delta_cb=None, generation: int = 0, elapsed_ms: int = 0) -> str:
        """Send an out-of-band translation request and wait for result."""
        await self._ensure_connected()

        context = self._build_context()
        user_content = f"{context}\n\nTranslate: {text}" if context else text

        tracker = _ResponseTracker(
            label=label, source=text,
            delta_cb=delta_cb, generation=generation,
            elapsed_ms=elapsed_ms, translator=self,
        )

        async with self._send_lock:
            await self._creation_queue.put(tracker)
            await self._ws.send(json.dumps({
                "type": "response.create",
                "response": {
                    "modalities": ["text"],
                    "instructions": self._build_instructions(),
                    "conversation": "none",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": user_content}]
                        }
                    ]
                }
            }))

        return await tracker.future

    async def translate_confirmed(self, sentence: str, elapsed_ms: int = 0):
        word_count = len(sentence.split())
        translated = await self._send_translation(
            sentence, f"CONFIRMED ({word_count}w)"
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
            self.confirmed_transcript = (self.confirmed_transcript + " " + sentence).strip() if self.confirmed_transcript else sentence
            self._update_summary_async()
            if self.on_confirmed:
                await self.on_confirmed(translated, elapsed_ms)

    async def translate_partial(self, text: str, elapsed_ms: int = 0):
        self._partial_generation += 1
        gen = self._partial_generation
        word_count = len(text.split())
        translated = await self._send_translation(
            text, f"PARTIAL ({word_count}w)",
            delta_cb=self.on_partial_delta, generation=gen, elapsed_ms=elapsed_ms,
        )
        if translated and gen == self._partial_generation:
            self.translated_partial = translated
            if self.on_partial:
                await self.on_partial(self.translated_partial, elapsed_ms)

    async def close(self):
        if self._reader_task:
            self._reader_task.cancel()
        if self._ws:
            await self._ws.close()
        self._connected = False
