import os
import asyncio
from openai import AsyncOpenAI

oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DETECT_PROMPT = """Analyze this English transcript from a live stream/video and determine the speaker's tone and register.

TRANSCRIPT:
{text}

Based on the language style, choose exactly ONE of these Korean speech levels that would best match the speaker's tone:

1. casual - 해체/반말 (friends talking, gaming streams, very relaxed)
   Use when: slang, filler words, addressing chat directly, cursing, incomplete sentences
   
2. casual_polite - 해요체 (friendly but polite, most YouTube content)
   Use when: conversational but structured, educational but approachable
   
3. formal - 합니다체 (news, lectures, business presentations)
   Use when: professional vocabulary, structured speech, formal setting
   
4. narrative - 하다체 (documentaries, storytelling, essays)
   Use when: descriptive, third person, explaining concepts with authority

Respond with ONLY the tone name (casual, casual_polite, formal, or narrative). Nothing else."""

TONE_INSTRUCTIONS = {
    "casual": (
        "Use casual Korean (해체/반말). Examples: ~해, ~했어, ~할게, ~인데, ~거든, ~잖아, ~임, ~ㅋㅋ. "
        "Sound natural like talking to friends or streaming. No formal endings."
    ),
    "casual_polite": (
        "Use casual polite Korean (해요체). Examples: ~해요, ~했어요, ~할 거예요, ~이에요. "
        "Friendly but polite tone."
    ),
    "formal": (
        "Use formal polite Korean (합니다체). Examples: ~합니다, ~했습니다, ~하겠습니다. "
        "Maintain professional, respectful tone throughout."
    ),
    "narrative": (
        "Use written/narrative Korean (하다체). Examples: ~한다, ~했다, ~할 것이다, ~이다. "
        "Maintain a descriptive, storytelling tone."
    ),
}


class ToneDetector:
    def __init__(self):
        self.word_buffer: list[str] = []
        self.current_tone = "casual_polite"  # Default
        self.detected = False
        self._detecting = False
        self._detect_task: asyncio.Task | None = None

    def feed_text(self, text: str):
        """Feed transcript text. Triggers detection after ~50 words."""
        if self.detected:
            return

        words = text.split()
        self.word_buffer = words  # Keep latest full partial

        if len(self.word_buffer) >= 30 and not self._detecting:
            self._detecting = True
            self._detect_task = asyncio.create_task(self._detect())

    async def _detect(self):
        text = " ".join(self.word_buffer[-100:])
        try:
            response = await oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": DETECT_PROMPT.format(text=text)},
                ],
                temperature=0,
                max_tokens=10,
            )

            result = response.choices[0].message.content.strip().lower()

            if result in TONE_INSTRUCTIONS:
                old = self.current_tone
                self.current_tone = result
                self.detected = True
                print(f"🎭 Tone detected: {old} → {result} (from {len(self.word_buffer)}w)")
            else:
                print(f"🎭 Tone detection unclear: '{result}', keeping {self.current_tone}")
                self._detecting = False  # Retry later

        except Exception as e:
            print(f"🎭 Tone detection error: {e}")
            self._detecting = False

    def get_tone_instruction(self) -> str:
        return TONE_INSTRUCTIONS.get(self.current_tone, TONE_INSTRUCTIONS["casual_polite"])