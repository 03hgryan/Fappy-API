"""
Phase 5 — Translation Engine (revision-aware)

Goal: Translate segments, not raw text.

Input:
{
  "segment_id": "seg-12",
  "stable": false,
  "text": "we should probably go to"
}

Output:
{
  "segment_id": "seg-12",
  "translation": "아마 우리는 …에 가야 할 것 같아",
  "revision": 3
}

Key idea:
- Translation is overwriteable
- UI must accept corrections
- No "append-only" text
"""


class TranslationEngine:
    def __init__(self, target_language: str = "ko"):
        """
        Args:
            target_language: Target language code (default: Korean)
        """
        self.target_language = target_language
        self.translation_cache = {}  # Cache by segment_id

    def translate(self, segment: dict) -> dict:
        """
        Translate a segment, respecting revisions.

        Args:
            segment: dict with segment_id, text, stable, revision

        Returns:
            dict with segment_id, translation, revision
        """
        segment_id = segment["segment_id"]
        text = segment["text"]
        revision = segment.get("revision", 0)

        # TODO: Implement actual translation logic
        # - Use translation API (Google, DeepL, etc.)
        # - Cache translations by segment_id
        # - Only retranslate if revision changed

        return {
            "segment_id": segment_id,
            "translation": f"[TODO: translate '{text}']",
            "revision": revision
        }

    def translate_batch(self, segments: list) -> list:
        """
        Translate multiple segments.

        Args:
            segments: List of segment dicts

        Returns:
            List of translation dicts
        """
        return [self.translate(seg) for seg in segments]
