import asyncio
from difflib import SequenceMatcher


def similarity(a: str, b: str) -> float:
    """Character-level similarity ratio between two strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def find_anchor_and_merge(combined: str, new_translation: str, threshold: float = 0.7) -> str:
    """
    Find the overlap between the tail of combined and the beginning of new_translation.
    
    Strategy:
    - Take progressively larger prefixes of new_translation (the overlap candidate)
    - Compare against windows in combined (searching deeper, last 40 words)
    - Find the best match, preferring LONGER matches over short ones
    - Keep combined up to anchor, replace with new_translation from that point
    """
    if not combined:
        return new_translation
    if not new_translation:
        return combined

    new_words = new_translation.split()
    combined_words = combined.split()

    if not new_words or not combined_words:
        return combined + " " + new_translation

    # If combined is very short, the new translation likely contains
    # everything the old one had (plus more). Just replace.
    if len(combined_words) <= 5:
        return new_translation

    best_score = 0.0
    best_combined_cut = len(combined_words)  # Default: keep all of combined
    best_new_start = 0  # Default: append all of new
    best_match_len = 0

    # Search last 40 words of combined
    search_start = max(0, len(combined_words) - 40)

    # Try different prefix lengths of new_translation
    # Minimum 3 words to avoid false positives on common short phrases
    max_prefix = min(len(new_words), 15)

    for prefix_len in range(3, max_prefix + 1):
        new_prefix = " ".join(new_words[:prefix_len])

        # Search for this prefix in the tail of combined
        for win_start in range(search_start, len(combined_words)):
            # Try windows of similar length (prefix_len ± 2)
            for win_len in range(max(2, prefix_len - 2), min(prefix_len + 3, len(combined_words) - win_start + 1)):
                win_end = win_start + win_len
                if win_end > len(combined_words):
                    continue

                window = " ".join(combined_words[win_start:win_end])
                score = similarity(window, new_prefix)

                # Weighted score: prefer longer matches
                # A 5-word match at 0.8 is better than a 3-word match at 0.9
                weighted = score * (prefix_len / max_prefix)

                if score >= threshold and weighted > best_score:
                    best_score = weighted
                    best_combined_cut = win_start
                    best_match_len = prefix_len

    if best_match_len >= 3 and best_score > 0:
        keep = " ".join(combined_words[:best_combined_cut]).strip()

        if keep:
            merged = keep + " " + new_translation
        else:
            merged = new_translation

        return merged
    else:
        # No overlap found — just append
        return combined + " " + new_translation


class Combiner:
    def __init__(self, on_combined=None):
        """
        on_combined: async callback(full_text, seq)
        """
        self.on_combined = on_combined
        self.combined_text = ""
        self.seq = 0
        self.latest_seq = -1
        self._lock = asyncio.Lock()

    def feed_translation(self, english_source: str, translated_text: str):
        if not translated_text.strip():
            return None
        self.seq += 1
        seq = self.seq
        asyncio.get_event_loop().call_soon(
            lambda: asyncio.ensure_future(self._run(translated_text.strip(), seq))
        )

    async def _run(self, new_translation: str, seq: int):
        try:
            async with self._lock:
                if seq <= self.latest_seq:
                    return

                if not self.combined_text:
                    self.latest_seq = seq
                    self.combined_text = new_translation
                    print(f"🔗 [{seq}] (first) | {new_translation[:80]}")
                    if self.on_combined:
                        await self.on_combined(new_translation, seq)
                    return

                old = self.combined_text
                merged = find_anchor_and_merge(old, new_translation)

                self.latest_seq = seq
                self.combined_text = merged

            old_words = old.split()
            merged_words = merged.split()

            # Count kept words
            kept = 0
            for i in range(min(len(old_words), len(merged_words))):
                if old_words[i] == merged_words[i]:
                    kept = i + 1
                else:
                    break

            changed = " ".join(merged_words[kept:]) if kept < len(merged_words) else ""
            print(f"🔗 [{seq}] kept:{kept}w | +{changed[:60]}")
            print(f"    result: ...{merged[-80:]}")

            if self.on_combined:
                await self.on_combined(merged, seq)

        except Exception as e:
            print(f"Combiner error: {type(e).__name__}: {e}")