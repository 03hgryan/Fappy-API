"""
Phase 4 — Hypothesis Manager (backend core)

Goal: Turn noisy partials into stable narrative units.

This is belief convergence, not text processing.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import time


# Tunable constants
CONFIDENCE_LOCK_THRESHOLD = 0.85
SILENCE_LOCK_MS = 600  # milliseconds


@dataclass
class SegmentState:
    """
    Internal state for a single utterance segment.

    This is non-negotiable — raw dicts won't scale.
    """
    segment_id: str
    text: str
    revision: int
    confidence: float
    is_stable: bool
    last_update_ts: float  # Unix timestamp in seconds

    def to_dict(self) -> dict:
        """Convert to dict for external use"""
        return {
            "segment_id": self.segment_id,
            "text": self.text,
            "revision": self.revision,
            "confidence": self.confidence,
            "stable": self.is_stable
        }


class HypothesisManager:
    """
    Manages ASR hypotheses using stability heuristics.

    Stability signals:
    - Signal A: confidence >= CONFIDENCE_LOCK_THRESHOLD
    - Signal B: silence_timeout exceeded
    - Signal C: is_final flag (if ASR provides it)

    Locking rule: ANY of these = stable
    """

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_LOCK_THRESHOLD,
        silence_lock_ms: float = SILENCE_LOCK_MS
    ):
        self.segments: Dict[str, SegmentState] = {}
        self.confidence_threshold = confidence_threshold
        self.silence_lock_ms = silence_lock_ms / 1000  # Convert to seconds

    def ingest(self, partial: dict) -> List[dict]:
        """
        Process incoming hypothesis partial.

        Args:
            partial: dict with segment_id, text, confidence, revision
                    Optional: is_final (from ASR engine)

        Returns:
            Reconciled segments ready for translation
        """
        segment_id = partial["segment_id"]
        now = time.time()

        # Step 1 — Fetch or create segment
        if segment_id not in self.segments:
            seg = SegmentState(
                segment_id=segment_id,
                text=partial["text"],
                revision=partial["revision"],
                confidence=partial["confidence"],
                is_stable=False,
                last_update_ts=now
            )
            self.segments[segment_id] = seg
        else:
            seg = self.segments[segment_id]

            # Step 2 — Reject outdated revisions
            if partial["revision"] <= seg.revision:
                return self.reconcile()

            # Step 3 — Update belief
            seg.text = partial["text"]
            seg.revision = partial["revision"]
            seg.confidence = partial["confidence"]
            seg.last_update_ts = now

        # Step 4 — Evaluate stability (Signal A and C)
        if not seg.is_stable:
            # Signal A: Confidence threshold
            if seg.confidence >= self.confidence_threshold:
                seg.is_stable = True

            # Signal C: Explicit final flag from ASR (if available)
            if partial.get("is_final", False):
                seg.is_stable = True

        return self.reconcile()

    def reconcile(self) -> List[dict]:
        """
        Return current state of all segments.

        Applies:
        - Signal B: Silence-based locking
        - Rule: Only ONE unstable segment at the end
        - Rule: Stable segments never change
        - Rule: Output order = narrative order

        Returns:
            List of segment dicts in narrative order
        """
        now = time.time()

        # Signal B — Silence-based locking (global logic)
        for seg in self.segments.values():
            if not seg.is_stable:
                time_since_update = now - seg.last_update_ts
                if time_since_update >= self.silence_lock_ms:
                    seg.is_stable = True

        # Enforce: Only ONE unstable segment at the end
        self._enforce_single_unstable()

        # Output in narrative order (sorted by segment_id)
        # Stable segments first, then the one unstable segment
        stable = []
        unstable = []

        for seg_id in sorted(self.segments.keys()):
            seg = self.segments[seg_id]
            if seg.is_stable:
                stable.append(seg.to_dict())
            else:
                unstable.append(seg.to_dict())

        return stable + unstable

    def _enforce_single_unstable(self):
        """
        Enforce Rule: Only ONE unstable segment at the end.

        If multiple segments are unstable, lock all but the most recent.
        This prevents unreadable UI.
        """
        unstable_segments = [
            seg for seg in self.segments.values()
            if not seg.is_stable
        ]

        if len(unstable_segments) <= 1:
            return  # Already satisfies rule

        # Sort by last_update_ts, keep most recent unstable
        unstable_segments.sort(key=lambda s: s.last_update_ts)

        # Lock all except the last one
        for seg in unstable_segments[:-1]:
            seg.is_stable = True

    def get_segment(self, segment_id: str) -> Optional[dict]:
        """Get specific segment by ID"""
        if segment_id in self.segments:
            return self.segments[segment_id].to_dict()
        return None

    def clear(self):
        """Clear all segments (useful for new session)"""
        self.segments.clear()
