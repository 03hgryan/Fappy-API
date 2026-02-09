import os
import json
import asyncio
from openai import AsyncOpenAI
from rapidfuzz import fuzz

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a real-time transcript segmenter. Given a buffer of unsegmented spoken words, decide if the beginning forms a translatable segment.

Rules:
- A translatable segment is a clause/phrase with enough meaning to translate on its own
- Make segments as long as possible while hitting a natural breakpoint
- Return words exactly as they appear
- If nothing is ready, return segment as empty string

Respond ONLY with JSON: {"segment": "...", "remaining": "..."}"""

USER_PROMPT = """Previous segments:
{context}

Buffer:
<buffer>{buffer}</buffer>

{flush}"""

MIN_WORDS = 4
FUZZY_THRESHOLD = 80


class Segmenter:
    def __init__(self, on_segment=None):
        self.on_segment = on_segment
        self.emitted_text = ""
        self.emitted_segments = []
        self.segment_index = 0
        self.seq = 0
        self.latest_resolved_seq = -1
        self._lock = asyncio.Lock()
        self._buffer = ""

    def feed_partial(self, partial_text: str, is_silent: bool):
        self._buffer = self._extract_buffer(partial_text)
        word_count = len(self._buffer.split()) if self._buffer else 0

        flush = is_silent and word_count > 0
        if flush or word_count >= MIN_WORDS:
            self.seq += 1
            return asyncio.create_task(self._segment(self.seq, self._buffer, flush))
        return None

    def _extract_buffer(self, partial: str) -> str:
        """Find unsegmented content by fuzzy-matching known text against partial."""
        known = (self.emitted_text + " " + self._buffer).strip() if self.emitted_text else self._buffer
        if not known:
            return partial.strip()

        known_words = known.split()
        partial_words = partial.split()

        # Try overlap lengths from longest to shortest.
        # Find longest suffix of known that fuzzy-matches a prefix of partial.
        max_check = min(len(known_words), len(partial_words))
        best_overlap = 0

        for length in range(max_check, 0, -1):
            suffix = " ".join(known_words[-length:])
            prefix = " ".join(partial_words[:length])
            if fuzz.ratio(suffix.lower(), prefix.lower()) >= FUZZY_THRESHOLD:
                best_overlap = length
                break  # Longest match found

        if best_overlap > 0:
            new_words = partial_words[best_overlap:]
            if new_words:
                return (self._buffer + " " + " ".join(new_words)).strip() if self._buffer else " ".join(new_words).strip()
            return self._buffer
        
        # No overlap — fallback
        print(f"⚠️  No overlap found, using full partial as buffer")
        return partial.strip()

    async def _segment(self, seq: int, buffer: str, flush: bool):
        if not buffer.strip():
            return

        context = "\n".join(f"- {s}" for s in self.emitted_segments[-5:]) or "(none)"
        flush_text = "Speaker paused. Emit the entire buffer as a segment." if flush else ""

        try:
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT.format(context=context, buffer=buffer, flush=flush_text)},
                ],
                temperature=0,
                max_tokens=300,
            )

            raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)
            segment = (result.get("segment") or "").strip()
            remaining = (result.get("remaining") or "").strip()

            async with self._lock:
                if seq <= self.latest_resolved_seq:
                    return

                if segment:
                    self.latest_resolved_seq = seq
                    self.emitted_text = (self.emitted_text + " " + segment).strip()
                    self.emitted_segments.append(segment)
                    self._buffer = remaining
                    self.segment_index += 1

                    print(f"✂️  SEGMENT #{self.segment_index}: {segment}")
                    print(f"    remaining: '{self._buffer}'")

                    if self.on_segment:
                        await self.on_segment(segment, self.segment_index)

        except json.JSONDecodeError as e:
            print(f"Segmenter JSON error: {e} | raw: {raw[:200]}")
        except Exception as e:
            print(f"Segmenter error: {type(e).__name__}: {e}")

    async def flush(self):
        async with self._lock:
            if self._buffer.strip():
                self.segment_index += 1
                segment = self._buffer.strip()
                self.emitted_text = (self.emitted_text + " " + segment).strip()
                self.emitted_segments.append(segment)
                self._buffer = ""

                print(f"✂️  SEGMENT #{self.segment_index} (flush): {segment}")
                if self.on_segment:
                    await self.on_segment(segment, self.segment_index)