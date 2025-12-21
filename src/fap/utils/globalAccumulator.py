"""
Global segment accumulator for immutable transcript history.

This is Layer 3 of the streaming ASR architecture:
- Layer 1: StreamingASR (2s rolling buffer, produces hypotheses)
- Layer 2: SegmentManager (finds stable prefixes within segments)
- Layer 3: GlobalAccumulator (append-only history, never changes)

Key invariant: Once text enters the accumulator, it NEVER changes.
"""

import time


class GlobalAccumulator:
    """
    Maintains append-only history of finalized segments.

    This is the "source of truth" for the transcript.
    Rolling buffer resets don't affect this - only silence detection does.
    """

    def __init__(self):
        """Initialize empty accumulator."""
        self.segments = []
        self.counter = 0  # Monotonic segment index for stable ordering

    def append(self, segment_id: str | None, text: str) -> dict:
        """
        Append a finalized segment to immutable history.

        Args:
            segment_id: Unique segment identifier (auto-generated if None)
            text: Finalized text (committed only, no partial)

        Returns:
            Finalized segment dict
        """
        if not text:
            return None

        # Guard: auto-generate segment_id if None
        if segment_id is None:
            segment_id = f"seg-auto-{int(time.time() * 1000)}"

        finalized = {
            "segment_index": self.counter,  # Monotonic index for stable ordering
            "segment_id": segment_id,
            "text": text,
            "timestamp_ms": int(time.time() * 1000),
            "final": True,
        }

        self.segments.append(finalized)
        self.counter += 1
        return finalized

    def get_full_transcript(self) -> str:
        """
        Get complete transcript from all finalized segments.

        Returns:
            Full transcript text
        """
        return " ".join(seg["text"] for seg in self.segments)

    def get_segments(self) -> list[dict]:
        """
        Get all finalized segments.

        Returns:
            List of finalized segment dicts
        """
        return self.segments.copy()

    def clear(self):
        """Clear all accumulated segments."""
        self.segments = []
