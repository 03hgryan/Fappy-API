# segmentManager.py
"""
Segment stability management using word-level timestamps.

Supports two ASR modes:
- "incremental" (Whisper): Words are prefix-stable, track candidates per region
- "rewriting" (Google): Hypothesis rewrites entirely, only trust is_final

Output Schema (the critical contract):
- finalized_words: Words that JUST became final this revision (append these!)
- stable_words: ALL finalized words so far in this segment
- unstable_words: Best guesses for words still in buffer (may change!)
- rendered_text: Convenience strings for display (NOT authoritative)

Invariants:
- Only SegmentManager decides when a word is final
- Final words are append-only (never change)
- Downstream code must NOT do string diffing
"""

from typing import TypedDict


class WordInfo(TypedDict):
    """A single word with timing and confidence."""
    word: str
    start_ms: int
    end_ms: int
    probability: float


class RenderedText(TypedDict):
    """Convenience rendering for display (NOT authoritative)."""
    stable: str      # Finalized text (won't change)
    unstable: str    # Preview text (may change)
    full: str        # stable + unstable


class SegmentOutput(TypedDict):
    """The output contract of SegmentManager.ingest()"""
    segment_id: str
    
    # 🔒 Words that JUST became final (this revision only)
    # Backend should APPEND these to permanent storage
    finalized_words: list[WordInfo]
    
    # 🧱 All finalized words so far in this segment
    stable_words: list[WordInfo]
    
    # 🌊 Best current hypothesis for words still in buffer
    # These are UNSTABLE and may change or disappear
    unstable_words: list[WordInfo]
    
    # 👁️ Convenience rendering (NOT authoritative - for display only)
    rendered_text: RenderedText
    
    revision: int
    final: bool


class SegmentManager:
    """
    Manages segment stability using word-level timestamps.

    Supports two ASR modes:
    - "incremental" (Whisper): Words are prefix-stable, track candidates per region
    - "rewriting" (Google): Hypothesis rewrites entirely, only trust is_final

    Core responsibilities:
    1. Group words by overlapping time regions (incremental mode)
    2. Track all candidate transcriptions per region (incremental mode)
    3. Select best candidate (highest probability) when locking (incremental mode)
    4. Replace unstable text until final (rewriting mode)
    5. Expose structured output with clear stability boundaries
    
    Downstream code should:
    - ONLY append finalized_words to permanent storage
    - ONLY use rendered_text for display
    - NEVER diff strings to determine what changed
    """

    def __init__(self, lock_margin_ms: int = 100, asr_mode: str = "incremental"):
        """
        Initialize segment manager.
        
        Args:
            lock_margin_ms: Extra margin when checking if word exited buffer
            asr_mode: "incremental" (Whisper) or "rewriting" (Google)
        """
        assert asr_mode in ("incremental", "rewriting"), f"Invalid asr_mode: {asr_mode}"
        
        self.lock_margin_ms = lock_margin_ms
        self.asr_mode = asr_mode
        
        # Word candidates grouped by audio region (incremental mode only)
        # Each region: {"start_ms", "end_ms", "candidates": [...]}
        self.word_regions: list[dict] = []
        
        # Words that have been locked (exited buffer, best candidate selected)
        self.locked_words: list[WordInfo] = []
        
        # Track segment ID for consistent identification
        self.segment_id: str | None = None
        
        print(f"🎛️ SegmentManager initialized: asr_mode={asr_mode}")

    def ingest(self, hypothesis: dict) -> SegmentOutput:
        """
        Process hypothesis, track word candidates, and emit structural stability updates.

        Args:
            hypothesis: Hypothesis dict from ASR containing:
                - segment_id: str
                - text: str (full transcript text)
                - words: list of word dicts with timing
                - start_time_ms: buffer start
                - end_time_ms: buffer end
                - revision: int
                - is_final: bool (for rewriting mode)

        Returns:
            SegmentOutput with clear stability boundaries
        """
        revision = hypothesis.get("revision", 0)
        segment_id = hypothesis.get("segment_id", "unknown")
        buffer_start_ms = hypothesis.get("start_time_ms", 0)
        buffer_end_ms = hypothesis.get("end_time_ms", buffer_start_ms + 2000)

        # Track segment ID on first ingest
        if self.segment_id is None:
            self.segment_id = segment_id

        # ============================================================
        # 🔁 REWRITING MODE (Google Cloud Speech)
        # ============================================================
        if self.asr_mode == "rewriting":
            return self._ingest_rewriting_mode(
                hypothesis, revision, segment_id, buffer_start_ms, buffer_end_ms
            )

        # ============================================================
        # 📝 INCREMENTAL MODE (Whisper / faster-whisper)
        # ============================================================
        return self._ingest_incremental_mode(
            hypothesis, revision, segment_id, buffer_start_ms, buffer_end_ms
        )

    # ================================================================
    # REWRITING MODE (Google Cloud Speech)
    # ================================================================

    def _ingest_rewriting_mode(
        self, 
        hypothesis: dict, 
        revision: int, 
        segment_id: str,
        buffer_start_ms: int,
        buffer_end_ms: int,
    ) -> SegmentOutput:
        """
        Handle Google-style rewriting ASR with time-based soft locking.
        
        Each hypothesis REPLACES the previous one entirely.
        Hard lock only on is_final=True.
        
        NEW: Time-based soft locking for progressive stability:
        - Track how long each word position has been stable
        - If a prefix of words is consistent for STABILITY_THRESHOLD_MS, soft-lock them
        - This gives progressive feedback like Whisper, while respecting Google's finals
        """
        import time
        
        text = hypothesis.get("text", "").strip()
        words = hypothesis.get("words", [])
        is_final = hypothesis.get("is_final", False)
        
        finalized_words: list[WordInfo] = []
        unstable_words: list[WordInfo] = []
        
        # Time-based stability settings
        current_time_ms = int(time.time() * 1000)
        STABILITY_THRESHOLD_MS = 800  # 1.5 seconds of consistency = soft lock
        
        # Initialize tracking attributes if needed
        if not hasattr(self, '_interim_word_history'):
            self._interim_word_history: list[tuple[str, int]] = []  # (word, first_seen_ms)
            self._soft_locked_count = 0
            self._final_processed = False  # Track if final was already processed

        # If final was already processed, ignore subsequent interims for same segment
        if not is_final and self._final_processed:
            # Check if this interim text matches or is subset of locked text
            current_locked_text = " ".join(w["word"] for w in self.locked_words).strip().lower()
            interim_text = text.strip().lower()
            if interim_text and current_locked_text and (interim_text in current_locked_text or current_locked_text in interim_text):
                print(f"📊 Revision {revision} (interim after final - SKIPPED)")
                stable_text = " ".join(w["word"] for w in self.locked_words)
                return {
                    "segment_id": self.segment_id or segment_id,
                    "finalized_words": [],
                    "stable_words": list(self.locked_words),
                    "unstable_words": [],
                    "rendered_text": {
                        "stable": stable_text,
                        "unstable": "",
                        "full": stable_text,
                    },
                    "revision": revision,
                    "final": False,
                }
        
        if is_final and words:
            # ✅ Final result with word timestamps - lock all words
            for w in words:
                word_info: WordInfo = {
                    "word": w.get("word", ""),
                    "start_ms": w.get("start_ms", buffer_start_ms),
                    "end_ms": w.get("end_ms", buffer_end_ms),
                    "probability": w.get("probability", 1.0),
                }
                finalized_words.append(word_info)

            # Build finalized text for comparison
            finalized_text = " ".join(w["word"] for w in finalized_words).strip().lower()
            current_locked_text = " ".join(w["word"] for w in self.locked_words).strip().lower()

            # Check if finalized text matches current locked text (soft-locked duplicates)
            if finalized_text == current_locked_text and self._soft_locked_count > 0:
                print(f"📊 Revision {revision} (FINAL - replacing soft-locked with real timestamps)")
                print(f"   🔄 Replacing {self._soft_locked_count} soft-locked words")
                # Clear soft-locked and replace with finalized (better timestamps)
                self.locked_words = []
                self.locked_words.extend(finalized_words)
                self.locked_words.sort(key=lambda w: w["start_ms"])
                # Reset interim tracking
                self._interim_word_history = []
                self._soft_locked_count = 0
                self._final_processed = True  # Mark final as processed
                print(f"   🔒 Hard-locked: {' '.join(w['word'] for w in finalized_words)}")
            elif finalized_text and current_locked_text and finalized_text in current_locked_text:
                # Finalized is a subset of what's already locked - skip to avoid duplication
                print(f"⚠️ Final text is subset of locked - skipping to avoid duplication")
                print(f"   Locked: \"{current_locked_text}\"")
                print(f"   Final:  \"{finalized_text}\"")
                finalized_words = []
                # Reset interim tracking anyway
                self._interim_word_history = []
                self._soft_locked_count = 0
                self._final_processed = True  # Mark final as processed
            elif finalized_words:
                # Normal case: replace soft-locked with finalized
                # Remove soft-locked words (they have approximate timestamps)
                if self._soft_locked_count > 0:
                    self.locked_words = self.locked_words[:-self._soft_locked_count] if self._soft_locked_count <= len(self.locked_words) else []

                # Add all finalized words
                self.locked_words.extend(finalized_words)
                self.locked_words.sort(key=lambda w: w["start_ms"])

                # Reset interim tracking
                self._interim_word_history = []
                self._soft_locked_count = 0
                self._final_processed = True  # Mark final as processed

                print(f"📊 Revision {revision} (FINAL)")
                print(f"   🔒 Hard-locked: {' '.join(w['word'] for w in finalized_words)}")
            
        elif is_final and text:
            # ✅ Final result but no word timestamps - create single entry
            # Remove soft-locked first
            if self._soft_locked_count > 0:
                self.locked_words = self.locked_words[:-self._soft_locked_count] if self._soft_locked_count <= len(self.locked_words) else []

            word_info: WordInfo = {
                "word": text,
                "start_ms": buffer_start_ms,
                "end_ms": buffer_end_ms,
                "probability": 1.0,
            }
            finalized_words = [word_info]
            self.locked_words.extend(finalized_words)
            self.locked_words.sort(key=lambda w: w["start_ms"])

            # Reset interim tracking
            self._interim_word_history = []
            self._soft_locked_count = 0
            self._final_processed = True  # Mark final as processed

            print(f"📊 Revision {revision} (FINAL, no word timestamps)")
            print(f"   🔒 Locked: \"{text}\"")
            
        else:
            # ⏳ Interim result - apply time-based soft locking
            current_words = text.split() if text else []
            
            # Update word history - track when each position was first seen
            new_history: list[tuple[str, int]] = []
            for i, word in enumerate(current_words):
                if i < len(self._interim_word_history):
                    old_word, first_seen = self._interim_word_history[i]
                    # Normalize comparison (case-insensitive, ignore trailing punctuation for matching)
                    old_normalized = old_word.lower().rstrip('.,!?')
                    new_normalized = word.lower().rstrip('.,!?')
                    if old_normalized == new_normalized:
                        # Same word - keep original timestamp
                        new_history.append((word, first_seen))
                    else:
                        # Word changed - reset timestamp
                        new_history.append((word, current_time_ms))
                else:
                    # New position - start tracking
                    new_history.append((word, current_time_ms))
            
            self._interim_word_history = new_history
            
            # Find contiguous prefix of stable words
            soft_locked_count = 0
            for i, (word, first_seen) in enumerate(self._interim_word_history):
                age_ms = current_time_ms - first_seen
                if age_ms >= STABILITY_THRESHOLD_MS:
                    soft_locked_count = i + 1
                else:
                    break  # Must be contiguous from start
            
            # Soft-lock new words
            newly_soft_locked: list[WordInfo] = []
            if soft_locked_count > self._soft_locked_count:
                for i in range(self._soft_locked_count, soft_locked_count):
                    word = self._interim_word_history[i][0]
                    word_info: WordInfo = {
                        "word": word,
                        "start_ms": buffer_start_ms + (i * 200),  # Approximate
                        "end_ms": buffer_start_ms + ((i + 1) * 200),
                        "probability": 0.85,  # Mark as soft-locked
                    }
                    newly_soft_locked.append(word_info)
                    self.locked_words.append(word_info)
                
                self._soft_locked_count = soft_locked_count
                finalized_words = newly_soft_locked
                
                print(f"📊 Revision {revision} (interim, +{len(newly_soft_locked)} soft-locked)")
                print(f"   🔓 Soft-locked: {' '.join(w['word'] for w in newly_soft_locked)}")
            else:
                print(f"📊 Revision {revision} (interim)")
            
            # Unstable = words after soft-locked portion
            if soft_locked_count < len(current_words):
                unstable_text = " ".join(current_words[soft_locked_count:])
                if unstable_text:
                    unstable_words = [{
                        "word": unstable_text,
                        "start_ms": buffer_start_ms,
                        "end_ms": buffer_end_ms,
                        "probability": 0.5,
                    }]
                    print(f"   🌊 Preview: \"{unstable_text}\"")
        
        # Build output
        stable_text = " ".join(w["word"] for w in self.locked_words)
        unstable_text = " ".join(w["word"] for w in unstable_words)
        
        print(f"   🔒 Stable:   \"{stable_text}\"")
        print(f"   🌊 Unstable: \"{unstable_text}\"")
        
        return {
            "segment_id": self.segment_id or segment_id,
            "finalized_words": finalized_words,
            "stable_words": list(self.locked_words),
            "unstable_words": unstable_words,
            "rendered_text": {
                "stable": stable_text,
                "unstable": unstable_text,
                "full": " ".join(p for p in [stable_text, unstable_text] if p),
            },
            "revision": revision,
            "final": is_final,
        }

    # ================================================================
    # INCREMENTAL MODE (Whisper / faster-whisper)
    # ================================================================

    def _ingest_incremental_mode(
        self,
        hypothesis: dict,
        revision: int,
        segment_id: str,
        buffer_start_ms: int,
        buffer_end_ms: int,
    ) -> SegmentOutput:
        """
        Handle Whisper-style incremental ASR.
        
        Words are prefix-stable. Track candidates per region,
        lock when they exit the buffer.
        """
        new_words = hypothesis.get("words", [])

        # === STEP 0: No word-level info → fallback rendering only ===
        if not new_words:
            if self.locked_words or self.word_regions:
                stable_text = " ".join(w["word"] for w in self.locked_words)
                unstable_words = self._get_best_in_buffer_words()
                unstable_text = " ".join(w["word"] for w in unstable_words)
                return {
                    "segment_id": self.segment_id or segment_id,
                    "finalized_words": [],
                    "stable_words": list(self.locked_words),
                    "unstable_words": unstable_words,
                    "rendered_text": {
                        "stable": stable_text,
                        "unstable": unstable_text,
                        "full": " ".join(p for p in [stable_text, unstable_text] if p),
                    },
                    "revision": revision,
                    "final": False,
                }
            
            # True fallback: no words ever existed
            return {
                "segment_id": self.segment_id or segment_id,
                "finalized_words": [],
                "stable_words": [],
                "unstable_words": [],
                "rendered_text": {
                    "stable": "",
                    "unstable": hypothesis.get("text", ""),
                    "full": hypothesis.get("text", ""),
                },
                "revision": revision,
                "final": False,
            }

        # === STEP 1: Filter invalid words ===
        valid_words = self._filter_valid_words(new_words)

        # === STEP 2: Add word candidates ===
        for word in valid_words:
            self._add_word_candidate(word, revision)

        # === STEP 3: Lock regions that exited the buffer ===
        newly_locked = self._lock_exited_regions(buffer_start_ms)

        # === STEP 4: Get best candidates still in buffer ===
        unstable_words = self._get_best_in_buffer_words()

        # === STEP 5: Stable words = locked_words ===
        stable_words = list(self.locked_words)

        # === STEP 6: Build rendered views (NON-authoritative) ===
        stable_text = " ".join(w["word"] for w in stable_words)
        unstable_text = " ".join(w["word"] for w in unstable_words)
        full_text = " ".join(
            part for part in [stable_text, unstable_text] if part
        )

        # === LOGGING ===
        print(f"📊 Revision {revision}")
        print(f"   Buffer: [{buffer_start_ms}–{buffer_end_ms}ms]")
        if newly_locked:
            print(f"   🔒 Newly locked: {' '.join(w['word'] for w in newly_locked)}")
        print(f"   🔒 Stable:   \"{stable_text}\"")
        print(f"   🌊 Unstable: \"{unstable_text}\"")

        return {
            "segment_id": self.segment_id or segment_id,
            "finalized_words": newly_locked,
            "stable_words": stable_words,
            "unstable_words": unstable_words,
            "rendered_text": {
                "stable": stable_text,
                "unstable": unstable_text,
                "full": full_text,
            },
            "revision": revision,
            "final": False,
        }

    # ================================================================
    # FINALIZE
    # ================================================================

    def finalize(self) -> SegmentOutput | None:
        """
        Finalize the segment (call on silence or disconnect).
        
        Locks all remaining regions and returns final output.
        """
        if self.asr_mode == "rewriting":
            # Rewriting mode: just return current locked state
            if not self.locked_words:
                return None

            stable_text = " ".join(w["word"] for w in self.locked_words)
            
            print(f"📊 Finalize (rewriting mode)")
            print(f"   ✅ Final transcript: \"{stable_text}\"")
            
            output: SegmentOutput = {
                "segment_id": self.segment_id or f"seg-final-{id(self)}",
                "finalized_words": [],  # Already finalized via is_final
                "stable_words": list(self.locked_words),
                "unstable_words": [],
                "rendered_text": {
                    "stable": stable_text,
                    "unstable": "",
                    "full": stable_text,
                },
                "revision": -1,
                "final": True,
            }
            
            # Reset state (including time-based tracking)
            self.locked_words = []
            self.segment_id = None
            if hasattr(self, '_interim_word_history'):
                self._interim_word_history = []
                self._soft_locked_count = 0
                self._final_processed = False

            return output

        # Incremental mode: lock all remaining regions
        final_locked: list[WordInfo] = []
        for region in self.word_regions:
            if region["candidates"]:
                best = max(region["candidates"], key=lambda c: c["probability"])
                word_info: WordInfo = {
                    "word": best["word"],
                    "start_ms": best["start_ms"],
                    "end_ms": best["end_ms"],
                    "probability": best["probability"],
                }
                self.locked_words.append(word_info)
                final_locked.append(word_info)
        
        self.word_regions = []
        
        if not self.locked_words:
            return None
        
        # Sort by time
        self.locked_words.sort(key=lambda w: w["start_ms"])
        final_locked.sort(key=lambda w: w["start_ms"])
        
        # Build final text
        stable_text = " ".join(w["word"] for w in self.locked_words)
        
        # Log final state
        print(f"📊 Finalize (incremental mode)")
        if final_locked:
            print(f"   🏁 FINALIZED: \"{' '.join(w['word'] for w in final_locked)}\"")
        print(f"   ✅ Final transcript: \"{stable_text}\"")
        
        output: SegmentOutput = {
            "segment_id": self.segment_id or f"seg-final-{id(self)}",
            "finalized_words": final_locked,
            "stable_words": list(self.locked_words),
            "unstable_words": [],
            "rendered_text": {
                "stable": stable_text,
                "unstable": "",
                "full": stable_text,
            },
            "revision": -1,
            "final": True,
        }
        
        # Reset state for next segment
        self.word_regions = []
        self.locked_words = []
        self.segment_id = None
        
        return output

    # ================================================================
    # INCREMENTAL MODE HELPERS
    # ================================================================

    def _filter_valid_words(self, words: list) -> list:
        """Filter out invalid words (zero duration, punctuation only)."""
        valid = []
        for w in words:
            duration = w["end_ms"] - w["start_ms"]
            word_text = w.get("word", "").strip()
            
            if duration <= 0:
                continue
            if not word_text:
                continue
            if word_text in ["-", ".", ",", "!", "?", "...", "—"]:
                continue
                
            valid.append(w)
        
        return valid

    def _get_overlap_threshold(self, word: dict) -> float:
        """Get overlap threshold based on word length."""
        word_text = word.get("word", "").strip(".,!?\"'")
        word_length = len(word_text)
        
        if word_length <= 3:
            return 0.20
        elif word_length <= 5:
            return 0.30
        elif word_length <= 7:
            return 0.40
        else:
            return 0.50

    def _add_word_candidate(self, word: dict, revision: int):
        """Add a word as a candidate to its matching region, or create new region."""
        matching_region = None
        for region in self.word_regions:
            if self._is_same_region(word, region):
                matching_region = region
                break
        
        if matching_region:
            matching_region["candidates"].append({
                "word": word["word"],
                "start_ms": word["start_ms"],
                "end_ms": word["end_ms"],
                "probability": word.get("probability", 0.5),
                "revision": revision,
            })
            matching_region["start_ms"] = min(matching_region["start_ms"], word["start_ms"])
            matching_region["end_ms"] = max(matching_region["end_ms"], word["end_ms"])
        else:
            self.word_regions.append({
                "start_ms": word["start_ms"],
                "end_ms": word["end_ms"],
                "candidates": [{
                    "word": word["word"],
                    "start_ms": word["start_ms"],
                    "end_ms": word["end_ms"],
                    "probability": word.get("probability", 0.5),
                    "revision": revision,
                }]
            })

    def _is_same_region(self, word: dict, region: dict) -> bool:
        """Check if word overlaps significantly with region."""
        overlap_start = max(word["start_ms"], region["start_ms"])
        overlap_end = min(word["end_ms"], region["end_ms"])
        overlap_duration = max(0, overlap_end - overlap_start)
        
        if overlap_duration == 0:
            return False
        
        word_duration = word["end_ms"] - word["start_ms"]
        region_duration = region["end_ms"] - region["start_ms"]
        shorter_duration = min(word_duration, region_duration)
        
        if shorter_duration <= 0:
            return False
        
        threshold = self._get_overlap_threshold(word)
        overlap_ratio = overlap_duration / shorter_duration
        return overlap_ratio > threshold

    def _lock_exited_regions(self, buffer_start_ms: int) -> list[WordInfo]:
        """Lock regions that have exited the buffer, selecting best candidate."""
        newly_locked: list[WordInfo] = []
        remaining_regions = []
        
        for region in self.word_regions:
            if region["end_ms"] < buffer_start_ms + self.lock_margin_ms:
                best = max(region["candidates"], key=lambda c: c["probability"])
                
                word_info: WordInfo = {
                    "word": best["word"],
                    "start_ms": best["start_ms"],
                    "end_ms": best["end_ms"],
                    "probability": best["probability"],
                }
                
                self.locked_words.append(word_info)
                newly_locked.append(word_info)
                
                if len(region["candidates"]) > 1:
                    candidates_str = ", ".join(
                        f"\"{c['word']}\"({c['probability']:.2f})" 
                        for c in region["candidates"]
                    )
                    print(f"   🏆 Selected \"{best['word']}\" from candidates: {candidates_str}")
            else:
                remaining_regions.append(region)
        
        self.word_regions = remaining_regions
        
        self.locked_words.sort(key=lambda w: w["start_ms"])
        newly_locked.sort(key=lambda w: w["start_ms"])
        
        return newly_locked

    def _get_best_in_buffer_words(self) -> list[WordInfo]:
        """Get the best candidate for each region still in buffer."""
        best_words: list[WordInfo] = []
        
        for region in self.word_regions:
            if region["candidates"]:
                best = max(region["candidates"], key=lambda c: c["probability"])
                best_words.append({
                    "word": best["word"],
                    "start_ms": best["start_ms"],
                    "end_ms": best["end_ms"],
                    "probability": best["probability"],
                })
        
        best_words.sort(key=lambda w: w["start_ms"])
        return best_words