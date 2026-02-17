import os
import json
import time
import asyncio
import websockets

SYSTEM_PROMPT = """You are a real-time subtitle translator. Translate the given text to {lang}.

Rules:
- Translate naturally, not word-by-word
- Match the speaker's energy and intent
- Output ONLY the translation, nothing else
- No quotes, no explanations, no labels"""


class _ResponseTracker:
    __slots__ = ("label", "source", "future", "text", "ttft", "start_time")

    def __init__(self, label: str, source: str):
        self.label = label
        self.source = source
        self.future: asyncio.Future = asyncio.get_event_loop().create_future()
        self.text = ""
        self.ttft: float | None = None
        self.start_time = time.monotonic()


class RealtimeTranslator:
    """Translator using OpenAI Realtime API over persistent WebSocket."""

    def __init__(self, on_confirmed=None, on_partial=None, tone_detector=None, target_lang=None):
        self.on_confirmed = on_confirmed
        self.on_partial = on_partial
        self.tone_detector = tone_detector
        self.target_lang = target_lang or "Korean"
        self.translated_confirmed = ""
        self.translated_partial = ""
        self.context_pairs: list[tuple[str, str]] = []

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
                "temperature": 0.3,
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
        if not self.context_pairs:
            return ""
        lines = []
        for source, translation in self.context_pairs:
            lines.append(f"Source: {source}\nTranslation: {translation}")
        return "Previous context:\n" + "\n\n".join(lines)

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
                        tracker.text += event.get("delta", "")

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

    async def _send_translation(self, text: str, label: str) -> str:
        """Send an out-of-band translation request and wait for result."""
        await self._ensure_connected()

        context = self._build_context()
        user_content = f"{context}\n\nTranslate: {text}" if context else text

        tracker = _ResponseTracker(label=label, source=text)

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
            sentence, f"CONFIRMED ({word_count}w, {len(self.context_pairs)}ctx)"
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
            if self.on_confirmed:
                await self.on_confirmed(translated, elapsed_ms)

    async def translate_partial(self, text: str, elapsed_ms: int = 0):
        word_count = len(text.split())
        translated = await self._send_translation(
            text, f"PARTIAL ({word_count}w, {len(self.context_pairs)}ctx)"
        )
        if translated:
            self.translated_partial = translated
            if self.on_partial:
                await self.on_partial(self.translated_partial, elapsed_ms)

    async def close(self):
        if self._reader_task:
            self._reader_task.cancel()
        if self._ws:
            await self._ws.close()
        self._connected = False
