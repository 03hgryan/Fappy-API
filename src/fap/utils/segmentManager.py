# segmentManager.py
"""
Segment stability management using word-level timestamps.

Key insight: Track ALL word candidates for each audio region,
then pick the BEST one (highest probability) when locking.

Strategy:
1. Group words by overlapping time regions
2. Use word-length-based overlap threshold (short words = lower threshold)
3. For each region, track all candidate transcriptions
4. When region exits buffer, lock the highest-probability version
5. Use locked words + current hypothesis for merged output
"""


class SegmentManager:
    """
    Manages segment stability using word-level timestamps.

    Instead of just keeping the latest word, we:
    - Track all word candidates for each audio region
    - Pick the best (highest probability) when locking
    - This prevents low-probability errors from being locked
    """

    def __init__(self, lock_margin_ms=100):
        """
        Initialize segment manager.
        
        Args:
            lock_margin_ms: Extra margin when checking if word exited buffer
        """
        self.lock_margin_ms = lock_margin_ms
        
        # Word candidates grouped by audio region
        self.word_regions = []  # List of {"start_ms", "end_ms", "candidates": [...]}
        
        # Words that have been locked (exited buffer, best candidate selected)
        self.locked_words = []

    def ingest(self, hypothesis: dict) -> dict:
        """
        Process hypothesis, track word candidates, and merge.

        Args:
            hypothesis: Hypothesis dict from StreamingASR

        Returns:
            Segment dict with committed text
        """
        text = hypothesis["text"]
        new_words = hypothesis.get("words", [])
        buffer_start_ms = hypothesis.get("start_time_ms", 0)
        buffer_end_ms = hypothesis.get("end_time_ms", buffer_start_ms + 2000)
        revision = hypothesis.get("revision", 0)
        
        if not new_words:
            return {
                "segment_id": hypothesis["segment_id"],
                "committed": text,
                "partial": "",
                "revision": revision,
                "final": False,
            }
        
        # === STEP 0: Filter out invalid words (zero duration, punctuation only) ===
        valid_words = self._filter_valid_words(new_words)
        
        # === STEP 1: Add new words to candidate regions ===
        for word in valid_words:
            self._add_word_candidate(word, revision)
        
        # === STEP 2: Lock regions that have exited the buffer ===
        newly_locked = self._lock_exited_regions(buffer_start_ms)
        
        # === STEP 3: Get current best words for regions still in buffer ===
        in_buffer_words = self._get_best_in_buffer_words(buffer_start_ms)
        
        # === STEP 4: Merge locked + in-buffer words ===
        all_words = self.locked_words + in_buffer_words
        all_words.sort(key=lambda w: w["start_ms"])
        
        merged_text = " ".join(w["word"] for w in all_words)
        
        # === LOGGING ===
        print(f"📊 Merge Result (revision {revision}):")
        print(f"   Buffer window: [{buffer_start_ms} - {buffer_end_ms}ms]")
        print(f"   New hypothesis: \"{text}\"")
        
        if newly_locked:
            locked_text = " ".join(w["word"] for w in newly_locked)
            print(f"   🔒 NEWLY LOCKED: \"{locked_text}\"")
        
        if self.locked_words:
            all_locked_text = " ".join(w["word"] for w in self.locked_words)
            locked_end = self.locked_words[-1]["end_ms"] if self.locked_words else 0
            print(f"   🔒 TOTAL LOCKED: \"{all_locked_text}\" [0 - {locked_end}ms]")
        else:
            print(f"   🔒 LOCKED: (nothing yet)")
            
        print(f"   ✅ MERGED: \"{merged_text}\"")
        
        # Debug: show candidate regions
        self._log_regions()
        
        return {
            "segment_id": hypothesis["segment_id"],
            "committed": merged_text,
            "partial": "",
            "revision": revision,
            "final": False,
        }

    def _filter_valid_words(self, words: list) -> list:
        """Filter out invalid words (zero duration, punctuation only)."""
        valid = []
        for w in words:
            duration = w["end_ms"] - w["start_ms"]
            word_text = w.get("word", "").strip()
            
            # Skip zero/negative duration
            if duration <= 0:
                continue
            
            # Skip empty words
            if not word_text:
                continue
            
            # Skip single punctuation
            if word_text in ["-", ".", ",", "!", "?", "...", "—"]:
                continue
                
            valid.append(w)
        
        return valid

    def _get_overlap_threshold(self, word: dict) -> float:
        """
        Get overlap threshold based on word length.
        
        Short words (like "Hey", "the", "it") need lower threshold
        because their timestamps vary more relative to their duration.
        
        Long words (like "something", "experienced") can use higher threshold.
        """
        word_text = word.get("word", "").strip(".,!?\"'")
        word_length = len(word_text)
        
        if word_length <= 3:       # "Hey", "the", "it", "be"
            return 0.20
        elif word_length <= 5:     # "When", "part", "ever"
            return 0.30
        elif word_length <= 7:     # "Vsauce", "Michael", "truly"
            return 0.40
        else:                      # "something", "experienced", "becomes"
            return 0.50

    def _add_word_candidate(self, word: dict, revision: int):
        """Add a word as a candidate to its matching region, or create new region."""
        # Find existing region that overlaps with this word
        matching_region = None
        for region in self.word_regions:
            if self._is_same_region(word, region):
                matching_region = region
                break
        
        if matching_region:
            # Add to existing region, update region bounds
            matching_region["candidates"].append({
                "word": word["word"],
                "start_ms": word["start_ms"],
                "end_ms": word["end_ms"],
                "probability": word.get("probability", 0.5),
                "revision": revision,
            })
            # Expand region bounds to cover all candidates
            matching_region["start_ms"] = min(matching_region["start_ms"], word["start_ms"])
            matching_region["end_ms"] = max(matching_region["end_ms"], word["end_ms"])
        else:
            # Create new region
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
        # Calculate overlap
        overlap_start = max(word["start_ms"], region["start_ms"])
        overlap_end = min(word["end_ms"], region["end_ms"])
        overlap_duration = max(0, overlap_end - overlap_start)
        
        if overlap_duration == 0:
            return False
        
        # Calculate durations
        word_duration = word["end_ms"] - word["start_ms"]
        region_duration = region["end_ms"] - region["start_ms"]
        shorter_duration = min(word_duration, region_duration)
        
        if shorter_duration <= 0:
            return False
        
        # Get threshold based on word length
        threshold = self._get_overlap_threshold(word)
        
        # Check if overlap is significant
        overlap_ratio = overlap_duration / shorter_duration
        return overlap_ratio > threshold

    def _lock_exited_regions(self, buffer_start_ms: int) -> list:
        """Lock regions that have exited the buffer, selecting best candidate."""
        newly_locked = []
        remaining_regions = []
        
        for region in self.word_regions:
            # Region has exited if its end is before buffer start (with margin)
            if region["end_ms"] < buffer_start_ms + self.lock_margin_ms:
                # Select best candidate by probability
                best_candidate = max(region["candidates"], key=lambda c: c["probability"])
                
                locked_word = {
                    "word": best_candidate["word"],
                    "start_ms": best_candidate["start_ms"],
                    "end_ms": best_candidate["end_ms"],
                    "probability": best_candidate["probability"],
                    "locked": True,
                }
                
                self.locked_words.append(locked_word)
                newly_locked.append(locked_word)
                
                # Log the selection
                if len(region["candidates"]) > 1:
                    candidates_str = ", ".join(
                        f"\"{c['word']}\"({c['probability']:.2f})" 
                        for c in region["candidates"]
                    )
                    print(f"   🏆 Selected \"{best_candidate['word']}\" from candidates: {candidates_str}")
            else:
                remaining_regions.append(region)
        
        self.word_regions = remaining_regions
        
        # Sort locked words by time
        self.locked_words.sort(key=lambda w: w["start_ms"])
        
        return newly_locked

    def _get_best_in_buffer_words(self, buffer_start_ms: int) -> list:
        """Get the best candidate for each region still in buffer."""
        best_words = []
        
        for region in self.word_regions:
            if region["candidates"]:
                # Get the highest probability candidate
                best = max(region["candidates"], key=lambda c: c["probability"])
                best_words.append({
                    "word": best["word"],
                    "start_ms": best["start_ms"],
                    "end_ms": best["end_ms"],
                    "probability": best["probability"],
                })
        
        best_words.sort(key=lambda w: w["start_ms"])
        return best_words

    def _log_regions(self):
        """Log current word regions for debugging."""
        if self.word_regions:
            print(f"   📍 Active regions: {len(self.word_regions)}")
            for i, region in enumerate(self.word_regions[:5]):  # Show first 5
                candidates = ", ".join(
                    f"\"{c['word']}\"({c['probability']:.2f})"
                    for c in region["candidates"][-3:]  # Show last 3 candidates
                )
                print(f"      Region {i}: [{region['start_ms']}-{region['end_ms']}ms] → {candidates}")

    def finalize(self) -> dict | None:
        """
        Finalize the segment (call on silence or disconnect).
        """
        # Lock all remaining regions
        for region in self.word_regions:
            if region["candidates"]:
                best = max(region["candidates"], key=lambda c: c["probability"])
                self.locked_words.append({
                    "word": best["word"],
                    "start_ms": best["start_ms"],
                    "end_ms": best["end_ms"],
                    "probability": best["probability"],
                    "locked": True,
                })
        
        self.word_regions = []
        
        if not self.locked_words:
            return None
        
        self.locked_words.sort(key=lambda w: w["start_ms"])
        final_text = " ".join(w["word"] for w in self.locked_words)
        
        result = {
            "segment_id": f"seg-final-{id(self)}",
            "committed": final_text,
            "partial": "",
            "revision": -1,
            "final": True,
        }
        
        # Reset state
        self.locked_words = []
        
        return result