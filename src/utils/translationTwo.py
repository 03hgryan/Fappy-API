"""
Streaming Translation Experimentation (translationTwo.py)

Strategy:
- Translate FULL ASR transcript every time
- Stream from GPT and log every chunk
- Use FUZZY MATCHING to find where previous translation ends in current
- Use EMBEDDINGS to find English source boundary matching confirmed Korean
- Extract and analyze only NEW content

NO frontend sending yet - pure data gathering.
"""

import os
import asyncio
from fastapi import WebSocket
import openai
from difflib import SequenceMatcher

from src.utils.embeddings import load_model, find_source_boundary

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def fuzzy_match(a: str, b: str) -> float:
    """Calculate fuzzy similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()


def split_by_punctuation(text: str) -> list[str]:
    """
    Split text into fragments by punctuation (commas and periods).
    Returns list of fragments WITH their trailing punctuation.
    
    Example: "Hello, world. How are you?" 
    → ["Hello,", "world.", "How are you?"]
    """
    if not text:
        return []
    
    fragments = []
    current = ""
    
    for char in text:
        current += char
        if char in ',.!?。！？、，；：':
            fragments.append(current.strip())
            current = ""
    
    # Add any remaining text
    if current.strip():
        fragments.append(current.strip())
    
    return fragments


def find_best_match_boundary(previous: str, current: str) -> tuple[int, float, str]:
    """
    Find where 'previous' text ends in 'current' text using sliding window fuzzy matching.
    
    Returns: (end_position, best_score, matched_fragment)
    - end_position: character index where previous content ends
    - best_score: similarity score of best match
    - matched_fragment: the fragment that matched
    """
    if not previous:
        return 0, 0.0, ""
    
    # Split current into fragments
    fragments = split_by_punctuation(current)
    
    best_score = 0.0
    best_end_pos = 0
    best_fragment = ""
    
    # Try all contiguous combinations
    for start_idx in range(len(fragments)):
        accumulated = ""
        current_pos = 0
        
        # Find where this fragment starts in the original text
        for i in range(start_idx):
            current_pos = current.find(fragments[i], current_pos) + len(fragments[i])
        
        for end_idx in range(start_idx + 1, len(fragments) + 1):
            # Build accumulated fragment
            fragment_text = " ".join(fragments[start_idx:end_idx])
            score = fuzzy_match(previous, fragment_text)
            
            if score > best_score:
                best_score = score
                # Find end position of this fragment combination
                end_pos = current.find(fragments[end_idx - 1], current_pos)
                if end_pos != -1:
                    best_end_pos = end_pos + len(fragments[end_idx - 1])
                    best_fragment = fragment_text
    
    return best_end_pos, best_score, best_fragment


def extract_sentence_at_index(text: str, sentence_index: int) -> str:
    """
    Extract the nth sentence (1-indexed) from text.
    Sentences end with . ! ? 。 ！ ？ or ... (ellipsis)
    """
    if not text:
        return ""
    
    # Replace ... with single character for easier parsing
    text = text.replace("...", "…")
    
    sentences = []
    current = ""
    
    for char in text:
        current += char
        if char in '.!?。！？…':
            sentences.append(current.strip())
            current = ""
    
    if sentence_index <= len(sentences):
        return sentences[sentence_index - 1]
    return ""


class StreamingTranslationWorker:
    """
    Experimental streaming translation with fuzzy matching.
    """
    
    def __init__(
        self,
        ws: WebSocket,
        source_lang: str = "English",
        target_lang: str = "Korean",
        model: str = "gpt-4o-mini",
        similarity_threshold: float = 0.85,
        debug: bool = True,  # Set to True for verbose logging
    ):
        self.ws = ws
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.model = model
        self.similarity_threshold = similarity_threshold
        self.debug = debug
        
        # Translation history
        self.previous_full_translation = ""
        self.current_full_translation = ""
        
        # Current stream state
        self.stream_chunks = []
        self.accumulated = ""
        
        # ASR tracking
        self.asr_update_count = 0
        self.current_transcript = ""  # Track current English transcript
        self.last_processed_transcript = ""  # Last transcript we actually processed
        
        # Confirmed translation tracking
        self.confirmed_korean = ""  # Accumulated confirmed Korean sentences (for embedding)
        self.matched_english = ""   # English portion that matches confirmed Korean
        self.remaining_english = "" # English portion to translate (not yet translated)
        
        # Pending: ONE sentence waiting for confirmation (needs 2 cycles)
        self.pending = ""           # The sentence text waiting for confirmation
        self.pending_cycles = 0     # How many ASR updates with punct we've seen
        
        # Shutdown
        self.shutdown_event = asyncio.Event()
        
        print(f"🧪 StreamingTranslationWorker initialized ({source_lang} → {target_lang}, debug={debug})")
    
    def start(self):
        if self.debug:
            print("▶️  Worker ready - fuzzy matching + embeddings mode")
        return None
    
    async def process_transcript(self, transcript: str, is_committed: bool = False):
        """Process every ASR update by streaming full translation."""
        if not transcript.strip():
            return
        
        # Skip if transcript is identical or nearly identical to last processed
        if self._is_minor_change(transcript):
            if self.debug:
                print(f"⏭️  Skipping minor change: \"{transcript[-30:]}\"")
            return
        
        self.asr_update_count += 1
        self.current_transcript = transcript  # Track current English transcript
        
        if self.debug:
            print("\n" + "─"*80)
            print(f"📥 ASR UPDATE #{self.asr_update_count} ({'COMMITTED' if is_committed else 'partial'})")
            print(f"   Full transcript: \"{transcript}\"")
            print("─"*80)
        
        # If we have confirmed Korean, find what English is already translated
        text_to_translate = transcript
        if self.confirmed_korean:
            # Use embedding to find boundary
            matched, remaining, score = find_source_boundary(
                transcript,
                self.confirmed_korean,
                threshold=0.50,
                tolerance=0.05,
                debug=self.debug,
            )
            
            self.matched_english = matched
            self.remaining_english = remaining
            
            if self.debug:
                print(f"\n   ✂️  CUTTING TRANSCRIPT:")
                print(f"      Confirmed Korean: \"{self.confirmed_korean[:50]}{'...' if len(self.confirmed_korean) > 50 else ''}\"")
                print(f"      Matched English: \"{matched}\"")
                print(f"      Remaining to translate: \"{remaining}\"")
            
            # Only translate the remaining part
            if remaining.strip():
                text_to_translate = remaining
            else:
                if self.debug:
                    print(f"      → Nothing new to translate!")
                return
        
        # Save previous translation
        self.previous_full_translation = self.current_full_translation
        
        # Stream translate only the NEW part
        if self.debug:
            print(f"\n   📤 TRANSLATING: \"{text_to_translate}\"")
        await self._stream_translate_full(text_to_translate)
        
        # Skip fuzzy matching analysis in production (it's just for logging)
        if self.debug:
            # TODO: COMBINED logic
            if self.confirmed_korean and text_to_translate != transcript:
                print(f"\n   🔗 [TODO] COMBINED:")
                print(f"      confirmed_korean: \"{self.confirmed_korean}\"")
                print(f"      accumulated: \"{self.accumulated}\"")
                print(f"      Need to combine without duplicates")
            
            # Analyze with fuzzy matching (debug only)
            self._analyze_with_fuzzy_matching()
        
        # Send to frontend
        await self._send_state()
        
        # Mark this transcript as processed
        self.last_processed_transcript = transcript
    
    def _is_minor_change(self, new_transcript: str) -> bool:
        """
        Check if new transcript is too similar to last processed to bother re-translating.
        Returns True if we should skip processing.
        
        Conservative approach: only skip if truly identical.
        """
        if not self.last_processed_transcript:
            return False
        
        # Only skip if completely identical
        if new_transcript == self.last_processed_transcript:
            return True
        
        return False
    
    async def _stream_translate_full(self, text: str):
        """Stream translate full text and log every chunk."""
        if not OPENAI_API_KEY:
            print("❌ OPENAI_API_KEY not configured")
            return
        
        client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Reset stream state
        self.stream_chunks = []
        self.accumulated = ""
        
        if self.debug:
            print(f"\n🔄 Starting GPT stream...")
            print("   Streaming chunks:")
        
        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"""Translate {self.source_lang} to {self.target_lang}. Output only the translation.

Important: 
- If the source text ends with sentence-ending punctuation (. ! ?), include the same type of punctuation in your translation. 
- If the source text does NOT end with sentence-ending punctuation, do NOT add any ending punctuation to your translation."""
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                stream=True,
                temperature=0.3,
                max_tokens=2000
            )
            
            chunk_num = 0
            prev_punct_count = 0
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    chunk_num += 1
                    content = chunk.choices[0].delta.content
                    self.stream_chunks.append(content)
                    self.accumulated += content
                    
                    # Count punctuation in accumulated
                    punct_count = self._count_punctuation(self.accumulated)
                    
                    # Log this chunk (debug only)
                    # if self.debug:
                    #     print(f"   [{chunk_num:3d}] +\"{content}\" → punct={punct_count}")
                    
                    # Send live update to frontend
                    await self._send_state()
                    
                    # Check if new sentence completed (just log, don't confirm yet)
                    if punct_count > prev_punct_count:
                        if self.debug:
                            new_sentence = extract_sentence_at_index(self.accumulated, punct_count)
                            print(f"\n   🔔 NEW SENTENCE #{punct_count}: \"{new_sentence}\"")
                        prev_punct_count = punct_count
            
            # Final result
            self.current_full_translation = self.accumulated
            final_punct_count = self._count_punctuation(self.current_full_translation)
            
            if self.debug:
                print(f"\n✅ Stream complete:")
                print(f"   Total chunks: {chunk_num}")
                print(f"   Final translation: \"{self.current_full_translation}\"")
                print(f"   Final punctuation count: {final_punct_count}")
            
            # === CONFIRMATION LOGIC (2 consecutive cycles) ===
            if final_punct_count > 0:
                # This stream has punctuation
                if self.pending:
                    self.pending_cycles += 1
                    if self.debug:
                        print(f"\n   🔄 Pending cycles: {self.pending_cycles}")
                    
                    if self.pending_cycles >= 2:
                        # Confirm! First sentence of this stream replaces old pending
                        first_sentence = extract_sentence_at_index(self.accumulated, 1)
                        self.confirmed_korean = (self.confirmed_korean + " " + first_sentence).strip()
                        if self.debug:
                            print(f"      ✅ CONFIRMED: \"{first_sentence}\" (replaces pending: \"{self.pending}\")")
                            print(f"      → confirmed_korean: \"{self.confirmed_korean}\"")
                        
                        # New pending = accumulated minus confirmed first sentence
                        remaining = self.accumulated[len(first_sentence):].strip()
                        if remaining and self._count_punctuation(remaining) > 0:
                            self.pending = remaining
                            self.pending_cycles = 1
                        else:
                            self.pending = ""
                            self.pending_cycles = 0
                        if self.debug:
                            print(f"      ⏳ pending: \"{self.pending}\" (cycles: {self.pending_cycles})")
                        
                        # Send confirmed update to frontend
                        await self._send_state()
                else:
                    # No pending yet, first sentence becomes pending
                    first_sentence = extract_sentence_at_index(self.accumulated, 1)
                    self.pending = first_sentence
                    self.pending_cycles = 1
                    if self.debug:
                        print(f"\n   ⏳ NEW pending: \"{self.pending}\" (cycles: {self.pending_cycles})")
            else:
                # No punctuation in this stream - reset cycles (must be consecutive)
                if self.pending_cycles > 0:
                    if self.debug:
                        print(f"\n   ⚠️ No punctuation - resetting cycles (was {self.pending_cycles})")
                    self.pending_cycles = 0
            
        except Exception as e:
            print(f"❌ Stream error: {e}")
            self.current_full_translation = ""
    
    def _count_punctuation(self, text: str) -> int:
        """Count sentence-ending punctuation marks (treating ... as one)."""
        if not text:
            return 0
        # Replace ... with single marker before counting
        text = text.replace("...", "…")
        return sum(1 for c in text if c in '.!?。！？…')
    
    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.
        Sentences end with . ! ? 。 ！ ？ or ... (ellipsis)
        Returns list of sentences WITH their ending punctuation.
        """
        if not text:
            return []
        
        # Replace ... with single character for easier parsing
        text = text.replace("...", "…")
        
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in '.!?。！？…':
                sentences.append(current.strip())
                current = ""
        
        # Add any remaining text (incomplete sentence)
        if current.strip():
            sentences.append(current.strip())
        
        return sentences
    
    def _analyze_embedding_boundary(self):
        """
        Use embeddings to find where English source matches confirmed Korean.
        Updates self.matched_english and self.remaining_english.
        """
        print(f"\n   📐 EMBEDDING BOUNDARY ANALYSIS:")
        print(f"      English transcript: \"{self.current_transcript[:60]}{'...' if len(self.current_transcript) > 60 else ''}\"")
        print(f"      Confirmed Korean: \"{self.confirmed_korean[:60]}{'...' if len(self.confirmed_korean) > 60 else ''}\"")
        
        # Find boundary using embeddings (prefix matching)
        matched_english, remaining_english, score = find_source_boundary(
            self.current_transcript,
            self.confirmed_korean,
            threshold=0.50,  # Lower threshold for testing
            tolerance=0.05,  # Allow slightly longer matches
        )
        
        # Store results
        self.matched_english = matched_english
        self.remaining_english = remaining_english
        
        print(f"\n      📊 RESULT:")
        print(f"         Matched English: \"{matched_english}\"")
        print(f"         Remaining (to translate): \"{remaining_english}\"")
        print(f"         Best score: {score:.3f}")
        
        # Stats
        total_words = len(self.current_transcript.split())
        matched_words = len(matched_english.split()) if matched_english else 0
        remaining_words = len(remaining_english.split()) if remaining_english else 0
        
        print(f"         Words: {matched_words} matched + {remaining_words} remaining = {total_words} total")
        if total_words > 0:
            savings_pct = (matched_words / total_words) * 100
            print(f"         Potential savings: {savings_pct:.1f}%")
        print()
    
    def _analyze_with_fuzzy_matching(self):
        """Use fuzzy matching to find where previous translation ends in current."""
        print(f"\n📊 FUZZY MATCHING ANALYSIS:")
        print(f"   Previous: \"{self.previous_full_translation}\"")
        print(f"   Current:  \"{self.current_full_translation}\"")
        
        if not self.previous_full_translation:
            print(f"   → First translation, nothing to compare")
            new_content = self.current_full_translation
            new_punct = self._count_punctuation(new_content)
            print(f"\n   📦 NEW CONTENT:")
            print(f"      Text: \"{new_content}\"")
            print(f"      Punctuation count: {new_punct}")
            return
        
        # Find best match boundary
        end_pos, score, matched_fragment = find_best_match_boundary(
            self.previous_full_translation,
            self.current_full_translation
        )
        
        print(f"\n   🔍 Best match:")
        print(f"      Score: {score:.3f} (threshold: {self.similarity_threshold})")
        print(f"      Matched fragment: \"{matched_fragment}\"")
        print(f"      End position: {end_pos}")
        
        if score >= self.similarity_threshold:
            # High confidence match - extract new content
            new_content = self.current_full_translation[end_pos:].strip()
            new_punct = self._count_punctuation(new_content)
            
            print(f"\n   ✅ HIGH CONFIDENCE MATCH")
            print(f"   📦 NEW CONTENT:")
            print(f"      Text: \"{new_content}\"")
            print(f"      Punctuation count: {new_punct}")
            
            if new_punct >= 2:
                print(f"\n   🎉 NEW CONTENT HAS 2+ PUNCTUATION!")
                print(f"      → Could confirm first sentence of new content")
        else:
            # Low confidence - might be a rewrite
            print(f"\n   ⚠️  LOW CONFIDENCE (score < {self.similarity_threshold})")
            print(f"      Previous and current translations differ significantly")
            print(f"      This might be a REWRITE or major change")
            
            # Still show "new content" as everything
            new_content = self.current_full_translation
            new_punct = self._count_punctuation(new_content)
            print(f"\n   📦 TREATING ENTIRE CURRENT AS NEW:")
            print(f"      Text: \"{new_content}\"")
            print(f"      Punctuation count: {new_punct}")
    
    async def _send_state(self):
        """Send current translation state to frontend."""
        try:
            # Combine confirmed + accumulated (excluding duplicate of confirmed)
            live = self.accumulated
            
            await self.ws.send_json({
                "type": "translation",
                "confirmed": self.confirmed_korean,
                "live": live,
            })
            
            # if self.debug:
            #     print(f"\n   📡 SENT TO FRONTEND:")
            #     print(f"      confirmed: \"{self.confirmed_korean[:50]}{'...' if len(self.confirmed_korean) > 50 else ''}\"")
            #     print(f"      live: \"{live[:50]}{'...' if len(live) > 50 else ''}\"")
        except Exception as e:
            print(f"   ❌ Failed to send to frontend: {e}")
    
    async def shutdown(self):
        if self.debug:
            print("\n" + "="*80)
            print("🛑 Shutting down StreamingTranslationWorker")
            print("="*80)
        self.shutdown_event.set()