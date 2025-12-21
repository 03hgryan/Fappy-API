"""
Segment stability management using N-revision prefix agreement.

Separates committed (stable) text from partial (unstable) text.
Stable text is locked and never revised.

Used in production ASR systems like Google Live Caption, Zoom captions,
Deepgram streaming, and AssemblyAI realtime.
"""


class SegmentManager:
    """
    Manages segment stability using N-revision prefix agreement.

    Key insight: Whisper is unstable at the tail, but stable at the head.
    We lock prefixes that appear unchanged in N consecutive hypotheses.
    """

    def __init__(self, stability_threshold: int = 3):
        """
        Initialize segment manager.

        Args:
            stability_threshold: Number of consecutive revisions needed for stability
        """
        self.stability_threshold = stability_threshold
        self.recent_hypotheses = []
        self.committed_text = ""

    def ingest(self, hypothesis: dict) -> dict:
        """
        Process hypothesis and determine stable vs partial text.

        Algorithm:
        1. Add hypothesis to sliding window
        2. Compute longest common prefix across recent hypotheses
        3. Commit only the new stable part (prevents duplication)

        Args:
            hypothesis: Hypothesis dict from StreamingASR

        Returns:
            Segment dict with committed and partial text:
            {
                "segment_id": "seg-123",
                "committed": "Michael Collins, ",
                "partial": "not the Irish Revolutionary",
                "revision": 12,
                "final": False
            }
        """
        text = hypothesis["text"]

        # Guard: detect rolling buffer discontinuity
        # If hypothesis doesn't start with committed text, buffer has moved forward
        if self.committed_text and not text.startswith(self.committed_text):
            # Reset state for implicit segment discontinuity
            self.recent_hypotheses = []
            self.committed_text = ""

        # Step 1: Add to sliding window
        self.recent_hypotheses.append(text)
        if len(self.recent_hypotheses) > self.stability_threshold:
            self.recent_hypotheses.pop(0)

        # Step 2: Find longest common prefix across recent hypotheses
        stable_prefix = self._common_prefix(self.recent_hypotheses)

        # Step 3: Commit only the new stable part (prevents duplication)
        if len(stable_prefix) > len(self.committed_text):
            self.committed_text = stable_prefix

        # Everything after committed text is partial
        partial = text[len(self.committed_text) :]

        return {
            "segment_id": hypothesis["segment_id"],
            "committed": self.committed_text,
            "partial": partial,
            "revision": hypothesis["revision"],
            "final": False,
        }

    def _common_prefix(self, strings: list[str]) -> str:
        """
        Find longest common prefix across all strings.

        Example:
            ["Michael Collins, not the",
             "Michael Collins, not the",
             "Michael Collins, not the Irish"]
            -> "Michael Collins, not the"

        Args:
            strings: List of strings to compare

        Returns:
            Longest common prefix
        """
        if not strings:
            return ""

        prefix = strings[0]
        for s in strings[1:]:
            while not s.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        return prefix

    def finalize(self) -> dict | None:
        """
        Finalize the segment (call on silence or disconnect).

        Returns:
            Final segment dict with final=True, or None if nothing to finalize
        """
        if not self.committed_text and not self.recent_hypotheses:
            return None

        # Use the most recent hypothesis as final text
        final_text = self.recent_hypotheses[-1] if self.recent_hypotheses else self.committed_text

        result = {
            "segment_id": f"seg-final-{id(self)}",
            "committed": final_text,
            "partial": "",
            "revision": -1,
            "final": True,
        }

        # Reset state
        self.recent_hypotheses = []
        self.committed_text = ""

        return result
