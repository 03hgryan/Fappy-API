"""
Sentence-by-sentence Translation Worker with Position Tracking.

Key insight: 
- Split transcript into sentences
- Track each sentence by its START POSITION in full transcript
- Same position = same sentence (even if text changed due to ASR revision)
- Different position = different sentence

Option 2: Show speculative translation immediately, confirm when stable
- Pending sentences (count < threshold): shown in yellow, may change
- Confirmed sentences (count >= threshold): shown in green, locked

Punctuation: If transcript lacks punctuation, call LLM to add it (debounced 500ms)

Output: 
{ 
    type: "sentences", 
    confirmed: [{source, translation}, ...],   # stable, won't change
    pending: [{source, translation}, ...],     # may still change
    remainder: {source, translation}           # incomplete sentence
}
"""

import os
import asyncio
import time
from fastapi import WebSocket
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def split_sentences(text: str) -> tuple[list[str], str]:
    """
    Split text into complete sentences and incomplete remainder.
    Returns: (complete_sentences, incomplete_remainder)
    """
    if not text.strip():
        return [], ""
    
    sentences = []
    current = ""
    text = text.strip()
    
    i = 0
    while i < len(text):
        current += text[i]
        
        if text[i] in '.!?':
            is_end = (i == len(text) - 1) or (i + 1 < len(text) and text[i + 1] == ' ')
            if is_end:
                sentences.append(current.strip())
                current = ""
                if i + 1 < len(text) and text[i + 1] == ' ':
                    i += 1
        
        i += 1
    
    return sentences, current.strip()


class TranslationWorker:
    """
    Position-based sentence tracking with immediate speculative display.
    """
    
    def __init__(
        self,
        ws: WebSocket,
        source_lang: str = "English",
        target_lang: str = "Korean",
        model: str = "gpt-4.1",
        stability_threshold: int = 3,
        punctuation_debounce: float = 0.5  # 500ms
    ):
        self.ws = ws
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.model = model
        self.stability_threshold = stability_threshold
        self.punctuation_debounce = punctuation_debounce
        
        # Position-based tracking: {position: {sentence, translation, count}}
        self.tracked = {}
        
        # Confirmed sentences (stable): [{source, translation, position}, ...]
        self.confirmed = []
        
        # Active translation tasks: {position: task}
        self.translation_tasks = {}
        
        # Remainder translation task
        self.remainder_task: asyncio.Task | None = None
        
        # Live translation (Track 2)
        self.live_task: asyncio.Task | None = None
        self.live_source = ""
        self.live_translation = ""
        
        # Punctuation debounce
        self.last_transcript = ""
        self.last_punctuated = ""
        self.punctuation_task: asyncio.Task | None = None
        self.last_process_time = 0
        
        # Shutdown
        self.shutdown_event = asyncio.Event()
        
        print(f"🔧 TranslationWorker initialized ({source_lang} → {target_lang}, threshold={stability_threshold})")
    
    def start(self):
        print("▶️  TranslationWorker started")
        return None
    
    async def process_transcript(self, transcript: str, is_committed: bool = False):
        """Process transcript - two tracks: sentences + live."""
        if not transcript.strip():
            return
        
        self.last_transcript = transcript
        current_time = asyncio.get_event_loop().time()
        
        # Track 1: Sentence processing (debounced punctuation)
        if self.punctuation_task is None or self.punctuation_task.done():
            if current_time - self.last_process_time >= self.punctuation_debounce:
                self.last_process_time = current_time
                self.punctuation_task = asyncio.create_task(
                    self._process_sentences(transcript, is_committed)
                )
        
        # Track 2: Live translation (always translate full text after confirmed)
        await self._update_live(transcript)
    
    async def _process_sentences(self, transcript: str, is_committed: bool):
        """Track 1: Punctuate and process for confirmed sentences."""
        try:
            punctuated = await self._add_punctuation(transcript)
            self.last_punctuated = punctuated
            
            print(f"📝 Punctuated: \"{punctuated[:80]}...\"" if len(punctuated) > 80 else f"📝 Punctuated: \"{punctuated}\"")
            
            await self._process_punctuated(punctuated, is_committed)
            
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"❌ Process error: {e}")
    
    async def _update_live(self, transcript: str):
        """Track 2: Translate everything after confirmed sentences."""
        # Find where confirmed text ends
        confirmed_end = 0
        if self.confirmed:
            last_confirmed = self.confirmed[-1]
            # Find end position of last confirmed sentence in transcript
            pos = transcript.lower().find(last_confirmed["source"].lower()[:30])
            if pos != -1:
                confirmed_end = pos + len(last_confirmed["source"])
        
        # Get live portion (everything after confirmed)
        live_source = transcript[confirmed_end:].strip()
        
        if not live_source:
            return
        
        # Throttle: only translate if previous task is done
        # (don't cancel - let it complete so user sees something)
        if self.live_task and not self.live_task.done():
            return  # Previous translation still running, skip this update
        
        # Start new live translation
        print(f"🔴 Live: \"{live_source[:50]}...\"" if len(live_source) > 50 else f"🔴 Live: \"{live_source}\"")
        self.live_task = asyncio.create_task(
            self._translate_live(live_source)
        )
    
    async def _translate_live(self, text: str):
        """Translate live portion."""
        try:
            translation = await self._translate(text)
            self.live_translation = translation
            await self._send_state()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"❌ Live translation error: {e}")
    
    async def _add_punctuation(self, text: str) -> str:
        """Call LLM to add punctuation."""
        if not OPENAI_API_KEY:
            return text
        
        client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        try:
            response = await client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Add sentence-ending punctuation (periods, question marks, exclamation points) to transcripts. Keep all words exactly the same. Output only the punctuated text."
                    },
                    {
                        "role": "user",
                        "content": "yeah I don't know what to do about it honestly like I was thinking maybe we could go but then again it's kind of far and I'm not sure if it's worth it what do you think"
                    },
                    {
                        "role": "assistant", 
                        "content": "Yeah, I don't know what to do about it honestly. Like I was thinking maybe we could go, but then again it's kind of far and I'm not sure if it's worth it. What do you think?"
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0,
                max_tokens=2000
            )
            result = response.choices[0].message.content.strip()
            print(f"🔤 Punctuation result: \"{result[:100]}...\"" if len(result) > 100 else f"🔤 Punctuation result: \"{result}\"")
            return result
        except Exception as e:
            print(f"⚠️ Punctuation error: {e}")
            return text
    
    async def _process_punctuated(self, transcript: str, is_committed: bool = False):
        """Process punctuated transcript using position-based tracking."""
        
        # Split into sentences
        sentences, remainder = split_sentences(transcript)
        
        # Find position of each sentence in full transcript
        current_sentences = []  # [(position, sentence), ...]
        search_start = 0
        for sentence in sentences:
            pos = transcript.find(sentence, search_start)
            if pos != -1:
                current_sentences.append((pos, sentence))
                search_start = pos + len(sentence)
        
        # Track and process each sentence
        current_positions = set()
        for pos, sentence in current_sentences:
            current_positions.add(pos)
            
            # Already confirmed at this position? Skip
            if any(c["position"] == pos for c in self.confirmed):
                continue
            
            # Track by position
            if pos in self.tracked:
                if self.tracked[pos]["sentence"] == sentence:
                    # Same position, same text → increment count
                    self.tracked[pos]["count"] += 1
                    count = self.tracked[pos]["count"]
                    short = sentence[:40] + "..." if len(sentence) > 40 else sentence
                    print(f"⏳ [{count}/{self.stability_threshold}] @{pos}: \"{short}\"")
                    
                    # Check if now stable
                    if count >= self.stability_threshold:
                        await self._promote_to_confirmed(pos)
                else:
                    # Same position, different text → ASR revision
                    old = self.tracked[pos]["sentence"]
                    print(f"🔄 Revision @{pos}: \"{old[:30]}\" → \"{sentence[:30]}\"")
                    
                    # Cancel old translation if running
                    if pos in self.translation_tasks:
                        self.translation_tasks[pos].cancel()
                        del self.translation_tasks[pos]
                    
                    # Reset tracking, start new translation
                    self.tracked[pos] = {"sentence": sentence, "translation": "", "count": 1}
                    self._start_translation(pos, sentence)
            else:
                # New position → start tracking and translating
                short = sentence[:40] + "..." if len(sentence) > 40 else sentence
                print(f"🆕 [1/{self.stability_threshold}] @{pos}: \"{short}\"")
                self.tracked[pos] = {"sentence": sentence, "translation": "", "count": 1}
                self._start_translation(pos, sentence)
        
        # Clean up disappeared positions
        disappeared = [p for p in list(self.tracked.keys()) if p not in current_positions]
        for p in disappeared:
            old_sentence = self.tracked[p]["sentence"]
            short = old_sentence[:30] + "..." if len(old_sentence) > 30 else old_sentence
            print(f"🗑️  Removed @{p}: \"{short}\"")
            
            # Cancel translation if running
            if p in self.translation_tasks:
                self.translation_tasks[p].cancel()
                del self.translation_tasks[p]
            
            del self.tracked[p]
        
        # Remove old remainder handling - live translation handles this now
        await self._send_state()
    
    def _start_translation(self, pos: int, sentence: str):
        """Start translating a sentence in background."""
        async def translate():
            try:
                translation = await self._translate(sentence)
                if pos in self.tracked and self.tracked[pos]["sentence"] == sentence:
                    self.tracked[pos]["translation"] = translation
                    print(f"✅ Translated @{pos}: \"{sentence[:30]}\" → \"{translation[:30]}\"")
                    await self._send_state()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"❌ Translation error @{pos}: {e}")
        
        self.translation_tasks[pos] = asyncio.create_task(translate())
    
    async def _promote_to_confirmed(self, pos: int):
        """Move a tracked sentence to confirmed."""
        if pos not in self.tracked:
            return
        
        data = self.tracked[pos]
        sentence = data["sentence"]
        translation = data["translation"]
        
        # If translation not ready yet, wait for it
        if not translation and pos in self.translation_tasks:
            try:
                await self.translation_tasks[pos]
                translation = self.tracked[pos]["translation"]
            except:
                translation = await self._translate(sentence)
        elif not translation:
            translation = await self._translate(sentence)
        
        print(f"🔒 Confirmed @{pos}: \"{sentence[:40]}\"")
        
        self.confirmed.append({
            "source": sentence,
            "translation": translation,
            "position": pos
        })
        self.confirmed.sort(key=lambda x: x["position"])
        
        # Clean up
        del self.tracked[pos]
        if pos in self.translation_tasks:
            del self.translation_tasks[pos]
        
        # Clear live translation since confirmed moved forward
        self.live_translation = ""
        await self._send_state()
    
    async def _send_state(self):
        """Send current state to frontend."""
        # Confirmed translations
        confirmed_translation = " ".join([c["translation"] for c in self.confirmed])
        
        await self.ws.send_json({
            "type": "translation",
            "confirmed": confirmed_translation,
            "live": self.live_translation
        })
        
        c_short = confirmed_translation[-40:] if len(confirmed_translation) > 40 else confirmed_translation
        l_short = self.live_translation[:40] if len(self.live_translation) > 40 else self.live_translation
        print(f"📤 Sent: confirmed=\"...{c_short}\" | live=\"{l_short}...\"")
    
    async def _translate(self, text: str) -> str:
        """Translate using OpenAI."""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")
        
        client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Context from confirmed sentences
        context = ""
        if self.confirmed:
            last_few = self.confirmed[-3:]
            context = "\n".join([f"{s['source']} → {s['translation']}" for s in last_few])
        
        system_prompt = f"Translate {self.source_lang} to {self.target_lang}. Output only the translation."
        if context:
            system_prompt += f"\n\nPrevious translations for context:\n{context}"
        
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        return response.choices[0].message.content.strip()
    
    async def shutdown(self):
        print("🛑 Shutting down...")
        self.shutdown_event.set()
        
        # Cancel punctuation task
        if self.punctuation_task and not self.punctuation_task.done():
            self.punctuation_task.cancel()
        
        # Cancel live translation task
        if self.live_task and not self.live_task.done():
            self.live_task.cancel()
        
        # Cancel all sentence translation tasks
        for task in self.translation_tasks.values():
            task.cancel()
        
        print("✅ Shutdown complete")