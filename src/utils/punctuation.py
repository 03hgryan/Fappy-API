import os
import re
import asyncio
from openai import AsyncOpenAI

oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

WORD_THRESHOLD = 15
TAIL_SKIP = 3  # skip last N words — they're unstable in STT partials

SYSTEM_PROMPT = "Split the incomplete text into two parts."

SPLIT_SCHEMA = {
    "type": "json_schema",
    "name": "split_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "part1": {"type": "string"},
            "part2": {"type": "string"},
        },
        "required": ["part1", "part2"],
        "additionalProperties": False,
    },
}

EXAMPLE_IN = "Some seem deeper like the uncanny feeling we get from recordings that make people from long ago"
EXAMPLE_OUT = '{"part1": "Some seem deeper", "part2": "like the uncanny feeling we get from recordings that make people from long ago"}'


def _strip_punct(w: str) -> str:
    return w.rstrip(".,?!")


class SentenceSplitter:
    """Async GPT-based sentence splitter for long unpunctuated STT streams."""

    def __init__(self):
        self._task: asyncio.Task | None = None
        self._pending = False
        self._split_at: int | None = None  # relative word index (0-based) to split after
        self._request_confirmed_count: int = 0
        self._request_words: list[str] = []  # snapshot (stripped) at request time
        self._request_len: int = 0

    def check(self, remaining_words: list[str], confirmed_count: int):
        """Called from feed(). Triggers GPT if remaining text is long and unpunctuated."""
        if self._pending or self._split_at is not None or len(remaining_words) < WORD_THRESHOLD:
            return
        text = " ".join(remaining_words)
        if re.search(r'[.?!]', text):
            return
        self._pending = True
        self._request_confirmed_count = confirmed_count
        self._request_words = [_strip_punct(w) for w in remaining_words]
        self._request_len = len(remaining_words)
        self._task = asyncio.create_task(self._find_split(list(remaining_words)))

    async def _find_split(self, words: list[str]):
        """Call GPT to find the best split position."""
        text = " ".join(words)
        try:
            response = await oai.responses.create(
                model="gpt-4.1-nano",
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": EXAMPLE_IN},
                    {"role": "assistant", "content": EXAMPLE_OUT},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_output_tokens=200,
                top_p=0,
                store=False,
                text={"format": SPLIT_SCHEMA},
            )

            result = response.output_text.strip()
            print(f"🔤 Splitter GPT: '{result}'")

            # Parse part1 from structured JSON response
            split_word_count = self._parse_split(result, words)
            if split_word_count is not None:
                # Apply TAIL_SKIP guard
                if split_word_count <= len(words) - TAIL_SKIP:
                    self._split_at = split_word_count - 1  # 0-indexed
                    part1 = " ".join(words[:split_word_count])
                    print(f"🔤 Split after word {split_word_count}: \"{part1}\"")
                else:
                    print(f"🔤 Split at {split_word_count} too close to tail, ignored")
            else:
                print("🔤 Could not parse split point")

        except Exception as e:
            print(f"🔤 Splitter error: {type(e).__name__}: {e}")
        finally:
            self._pending = False

    def _parse_split(self, result: str, words: list[str]) -> int | None:
        """Extract word count of part1 from structured JSON response."""
        import json
        try:
            data = json.loads(result)
            part1 = data.get("part1", "").strip()
        except (json.JSONDecodeError, AttributeError):
            return None

        if not part1:
            return None

        part1_words = part1.split()

        # Match part1 words against original words to find the split position
        if len(part1_words) == 0 or len(part1_words) >= len(words):
            return None

        # Verify the words match
        matched = 0
        for i, pw in enumerate(part1_words):
            if i >= len(words):
                break
            if _strip_punct(pw).lower() == _strip_punct(words[i]).lower():
                matched += 1

        if matched >= len(part1_words) * 0.8:  # allow small mismatches
            return len(part1_words)
        return None

    def take_split(self, current_confirmed_count: int, current_remaining: list[str]) -> int | None:
        """Return number of words to confirm, or None.

        Guards:
        1. No natural confirmation happened during GPT latency
        2. Current remaining text is at least as long as request snapshot
        3. Word at split position still matches
        """
        if self._split_at is None:
            return None
        if current_confirmed_count != self._request_confirmed_count:
            print(f"🔤 Split discarded (confirmed advanced {self._request_confirmed_count} → {current_confirmed_count})")
            self._split_at = None
            return None
        if len(current_remaining) < self._request_len:
            return None  # keep split, wait for full text

        rel_idx = self._split_at
        if rel_idx >= len(current_remaining):
            self._split_at = None
            return None
        if _strip_punct(current_remaining[rel_idx]).lower() != self._request_words[rel_idx].lower():
            print(f"🔤 Split discarded (word changed: '{self._request_words[rel_idx]}' → '{_strip_punct(current_remaining[rel_idx])}')")
            self._split_at = None
            return None

        word_count = rel_idx + 1
        self._split_at = None
        return word_count
