"""
Translation service using OpenAI's Responses API.
Translates only NEW content incrementally.
"""

import os
import asyncio
import re
from fastapi import WebSocket, WebSocketDisconnect
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class TranslationWorker:
    """Manages incremental translation with rolling cut-off"""
    
    def __init__(self, ws: WebSocket, confirmation_cycles: int = 5):
        self.ws = ws
        self.confirmation_cycles = confirmation_cycles
        self.shutdown_event = asyncio.Event()
        
        # Cut-off tracking
        self.translated_up_to = 0
        self.last_translated_sentence = ""
        self.pending_sentences = {}  # {sentence@pos: {count, start_pos}}
        
        # Context tracking
        self.confirmed_sentences = []  # Store confirmed sentences for context
        
        # Single active translation
        self.active_task = None
        self.last_translated_text = ""
        
        # Current state for frontend
        self.current_full_text = ""
        self.current_confirmed_text = ""
        self.current_pending_text = ""
        
        print(f"🔧 TranslationWorker initialized (confirmation_cycles={confirmation_cycles})")
    
    def start(self):
        print("▶️  TranslationWorker started")
        return None
    
    def _find_remaining_start(self, text: str) -> int:
        """Find where to start processing based on last confirmed sentence."""
        
        if not self.last_translated_sentence:
            return 0
        
        sentence = self.last_translated_sentence
        
        # Find all occurrences
        positions = []
        start = 0
        while True:
            pos = text.find(sentence, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        
        if not positions:
            print(f"⚠️  Last sentence not found, using position {self.translated_up_to}")
            return min(self.translated_up_to, len(text))
        
        if len(positions) == 1:
            return positions[0] + len(sentence)
        
        closest = min(positions, key=lambda p: abs(p - self.translated_up_to))
        return closest + len(sentence)
    
    async def process_transcript(self, text: str, is_committed: bool = False):
        """Process incoming transcript."""
        
        # Store full text
        self.current_full_text = text
        
        # 1. Get remaining text after cut-off
        remaining_start = self._find_remaining_start(text)
        remaining_text = text[remaining_start:].strip()
        
        # Update confirmed/pending text
        self.current_confirmed_text = text[:self.translated_up_to].strip()
        self.current_pending_text = remaining_text
        
        # Send transcript update to frontend
        await self._send_transcript_update(is_committed)
        
        if not remaining_text:
            return
        
        # 2. Update pending sentences for cut-off tracking
        await self._update_pending(text, remaining_start, is_committed)
        
        # 3. Recalculate remaining after potential cut-off move
        remaining_start = self._find_remaining_start(text)
        remaining_text = text[remaining_start:].strip()
        
        # Update state again after potential cut-off move
        self.current_confirmed_text = text[:self.translated_up_to].strip()
        self.current_pending_text = remaining_text
        
        if not remaining_text:
            return
        
        # 4. Translate if different from last and no active translation
        if remaining_text != self.last_translated_text:
            await self._translate_remaining(remaining_text)
    
    async def _send_transcript_update(self, is_final: bool = False):
        """Send transcript state to frontend."""
        message = {
            "type": "transcript_update",
            "data": {
                "full_text": self.current_full_text,
                "confirmed_text": self.current_confirmed_text,
                "pending_text": self.current_pending_text,
                "confirmed_up_to": self.translated_up_to,
                "confirmed_sentences": self.confirmed_sentences.copy(),
                "is_final": is_final
            }
        }
        await self.ws.send_json(message)
        
        # Log
        conf = self.current_confirmed_text
        pend = self.current_pending_text
        # conf = self.current_confirmed_text[:30] + "..." if len(self.current_confirmed_text) > 30 else self.current_confirmed_text
        # pend = self.current_pending_text[:30] + "..." if len(self.current_pending_text) > 30 else self.current_pending_text
        print(f"📤 → Frontend: transcript_update | confirmed: \"{conf}\" | pending: \"{pend}\"")
    
    async def _send_sentence_confirmed(self, sentence: str):
        """Notify frontend that a sentence was confirmed."""
        message = {
            "type": "sentence_confirmed",
            "data": {
                "sentence": sentence,
                "confirmed_up_to": self.translated_up_to,
                "total_confirmed": self.confirmed_sentences.copy()
            }
        }
        await self.ws.send_json(message)
        
        # Log
        short = sentence
        # short = sentence[:40] + "..." if len(sentence) > 40 else sentence
        print(f"📤 → Frontend: sentence_confirmed | \"{short}\"")
    
    async def _update_pending(self, text: str, remaining_start: int, is_committed: bool):
        """Track sentences for confirmation (cut-off point)."""
        
        remaining_text = text[remaining_start:].strip()
        if not remaining_text:
            return
        
        sentences = self._split_sentences(remaining_text)
        if not sentences:
            return
        
        # Determine completed sentences
        if remaining_text.rstrip().endswith(('.', '!', '?')):
            completed_sentences = sentences
        else:
            completed_sentences = sentences[:-1] if len(sentences) > 1 else []
        
        # Build (sentence, position) pairs
        current_pos = remaining_start
        completed_with_positions = []
        
        for sentence in sentences:
            pos = text.find(sentence, current_pos)
            if pos != -1:
                if sentence in completed_sentences:
                    completed_with_positions.append((sentence, pos))
                current_pos = pos + len(sentence)
        
        # Update counts
        current_pending_keys = set()
        
        for sentence, start_pos in completed_with_positions:
            key = f"{sentence}@{start_pos}"
            current_pending_keys.add(key)
            
            if key in self.pending_sentences:
                count = self.pending_sentences[key]["count"]
                if count < self.confirmation_cycles - 1:
                    self.pending_sentences[key]["count"] += 1
                    new_count = self.pending_sentences[key]["count"]
                    short = sentence
                    # short = sentence[:30] + "..." if len(sentence) > 30 else sentence
                    print(f"⏳ [{new_count}/{self.confirmation_cycles}] \"{short}\"")
                else:
                    # Confirmed! Move cut-off
                    sentence_end = start_pos + len(sentence)
                    self.translated_up_to = sentence_end
                    self.last_translated_sentence = sentence
                    del self.pending_sentences[key]
                    
                    # Add to confirmed sentences for context
                    self.confirmed_sentences.append(sentence)
                    
                    short = sentence
                    # short = sentence[:40] + "..." if len(sentence) > 40 else sentence
                    print(f"✅ Confirmed: \"{short}\" (cut-off: {sentence_end})")
                    
                    # Notify frontend
                    await self._send_sentence_confirmed(sentence)
                    
                    # Clear pending before this position
                    for k in list(self.pending_sentences.keys()):
                        if self.pending_sentences[k]["start_pos"] < sentence_end:
                            del self.pending_sentences[k]
                    
                    # Reset last translated to force new translation
                    self.last_translated_text = ""
            else:
                self.pending_sentences[key] = {
                    "count": 1,
                    "start_pos": start_pos
                }
                short = sentence
                # short = sentence[:30] + "..." if len(sentence) > 30 else sentence
                print(f"🆕 [1/{self.confirmation_cycles}] \"{short}\"")
        
        # Clean up disappeared
        disappeared = []
        for key in list(self.pending_sentences.keys()):
            if key not in current_pending_keys:
                sentence = key.rsplit("@", 1)[0]
                disappeared.append(sentence[:20])
                del self.pending_sentences[key]
        
        if disappeared:
            print(f"🗑️  Removed {len(disappeared)} disappeared")
        
        if is_committed:
            print(f"📨 Committed transcript received")
            self.pending_sentences.clear()
    
    async def _translate_remaining(self, text: str):
        """Translate remaining text. Skip if previous still running."""
        
        # If previous translation still running, skip this cycle
        if self.active_task and not self.active_task.done():
            return
        
        short = text
        # short = text[:50] + "..." if len(text) > 50 else text
        print(f"🔄 Translating: \"{short}\"")
        
        self.last_translated_text = text
        self.active_task = asyncio.create_task(
            self._do_translate(text)
        )
    
    async def _do_translate(self, text: str):
        """Perform translation with context."""
        try:
            # Get last 3 confirmed sentences as context
            context_sentences = self.confirmed_sentences[-3:] if self.confirmed_sentences else []
            
            await translate_and_send(
                self.ws,
                text,
                is_partial=True,
                context=context_sentences,
                confirmed_text=self.current_confirmed_text,
                pending_text=self.current_pending_text
            )
        except WebSocketDisconnect:
            print("❌ WebSocket disconnected")
        except Exception as e:
            print(f"❌ Translation error: {type(e).__name__}: {e}")
    
    async def shutdown(self):
        print("🛑 Shutdown requested")
        self.shutdown_event.set()
        
        # Wait for active translation to complete
        if self.active_task and not self.active_task.done():
            print("⏳ Waiting for active translation...")
            try:
                await asyncio.wait_for(self.active_task, timeout=5.0)
            except asyncio.TimeoutError:
                print("⚠️  Translation timed out")
        
        print("✅ Shutdown complete")
    
    def _split_sentences(self, text: str) -> list:
        parts = re.split(r'([.!?])\s+', text)
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            if i + 1 < len(parts):
                sentences.append(parts[i] + parts[i + 1])
            else:
                sentences.append(parts[i])
        if len(parts) % 2 == 1:
            sentences.append(parts[-1])
        return [s.strip() for s in sentences if s.strip()]


async def translate_and_send(
    ws: WebSocket,
    text: str,
    is_partial: bool = False,
    context: list = None,
    confirmed_text: str = "",
    pending_text: str = "",
    source_lang: str = "English",
    target_lang: str = "Korean",
    model: str = "gpt-4.1"
):
    """Translate text using OpenAI and stream to client."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not configured")
    
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    # Build system prompt with context
    if context:
        context_text = " ".join(context)
        system_prompt = f"""Translate {source_lang} to {target_lang}. Output only the translation.

For context, here are the previous sentences (already translated, do not include in output):
{context_text}

Maintain consistent tone and style with the context above."""
    else:
        system_prompt = f"Translate {source_lang} to {target_lang}. Output only the translation."
    
    stream = await client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        stream=True
    )

    translated = ""

    async for event in stream:
        if event.type == "response.output_text.delta":
            translated += event.delta
            
            await ws.send_json({
                "type": "translation_delta",
                "data": {
                    "delta": event.delta,
                    "text": translated,
                    "source_text": text,
                    "context": context or [],
                    "confirmed_source": confirmed_text,
                    "pending_source": pending_text,
                    "is_partial": is_partial
                }
            })
            
            # Log (only log every 10 characters to reduce noise)
            if len(translated) % 10 == 0 or len(translated) <= 3:
                short = translated
                # short = translated[:30] + "..." if len(translated) > 30 else translated
                print(f"📤 → Frontend: translation_delta | \"{short}\"")

        elif event.type in ["response.done", "response.completed"]:
            src_short = text
            trans_short = translated
            # src_short = text[:30] + "..." if len(text) > 30 else text
            # trans_short = translated[:30] + "..." if len(translated) > 30 else translated
            print(f"  ✓ [{src_short}] → [{translated}]")
            
            await ws.send_json({
                "type": "translation_complete",
                "data": {
                    "text": translated,
                    "source_text": text,
                    "context": context or [],
                    "confirmed_source": confirmed_text,
                    "pending_source": pending_text,
                    "is_partial": is_partial
                }
            })
            
            # Log
            print(f"📤 → Frontend: translation_complete | src: \"{src_short}\" → \"{trans_short}\"")
            break