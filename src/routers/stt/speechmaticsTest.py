import os
import json
import re
import base64
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from speechmatics.rt import (
    AsyncClient,
    ServerMessageType,
    TranscriptionConfig,
    TranscriptResult,
    OperatingPoint,
    AudioFormat,
    AudioEncoding,
)
from utils.translationExp import Translator
from utils.tone import ToneDetector

router = APIRouter()

SPEECHMATICS_API_KEY = os.getenv("SPEECHMATICS_API_KEY")


@router.websocket("")
async def stream(ws: WebSocket):
    await ws.accept()

    final_text = ""
    current_partial = ""
    prev_full = ""
    confirmed_text = ""
    confirmed_word_count = 0
    prev_remaining = ""
    partial_count = 0
    PARTIAL_INTERVAL = 2
    CONFIRM_PUNCT_COUNT = 1  # number of punctuation+text matches needed to confirm
    closed = False

    async def on_confirmed_translation(confirmed_korean):
        if not closed:
            await ws.send_json({"type": "combined", "full": confirmed_korean})

    async def on_partial_translation(partial_korean):
        if not closed:
            await ws.send_json({"type": "translation", "text": partial_korean})

    tone_detector = ToneDetector()
    translator = Translator(on_confirmed=on_confirmed_translation, on_partial=on_partial_translation, tone_detector=tone_detector)

    def get_full_text():
        if not final_text:
            return current_partial.strip()
        if not current_partial:
            return final_text.strip()
        f = final_text.rstrip()
        p = current_partial.strip()
        best = 0
        for length in range(1, min(len(f), len(p)) + 1):
            if f.endswith(p[:length]):
                best = length
        return (f + p[best:]).strip() if best > 0 else (f + " " + p).strip()

    def update_remaining(full):
        nonlocal confirmed_text, confirmed_word_count, prev_remaining, partial_count

        tone_detector.feed_text(full)
        words = full.split()
        remaining_text = " ".join(words[confirmed_word_count:])

        # Check for confirmed sentence: need CONFIRM_PUNCT_COUNT punctuation+text matches
        matches = list(re.finditer(r'[.?!]\s+\w', remaining_text))
        if len(matches) >= CONFIRM_PUNCT_COUNT:
            cut_match = matches[-CONFIRM_PUNCT_COUNT]
            cut = cut_match.start() + 1
            new_confirmed = remaining_text[:cut].strip()

            if new_confirmed:
                confirmed_text = (confirmed_text + " " + new_confirmed).strip() if confirmed_text else new_confirmed
                confirmed_word_count += len(new_confirmed.split())
                remaining_text = " ".join(words[confirmed_word_count:])
                print(f"✅ confirmed: \"{new_confirmed}\"")
                loop = asyncio.get_event_loop()
                loop.create_task(translator.translate_confirmed(new_confirmed))
                partial_count = 0

        # Fire partial translation every N updates
        partial_count += 1
        if partial_count % PARTIAL_INTERVAL == 0 and remaining_text:
            loop = asyncio.get_event_loop()
            loop.create_task(translator.translate_partial(remaining_text))

        if remaining_text != prev_remaining:
            prev_remaining = remaining_text

    client = AsyncClient(api_key=SPEECHMATICS_API_KEY)
    await client.__aenter__()

    @client.on(ServerMessageType.ADD_TRANSCRIPT)
    def on_final(msg):
        nonlocal final_text, prev_full
        transcript = TranscriptResult.from_message(msg).metadata.transcript
        if transcript:
            final_text = final_text + transcript if final_text else transcript
            full = get_full_text()
            if full != prev_full:
                prev_full = full
                update_remaining(full)

    @client.on(ServerMessageType.ADD_PARTIAL_TRANSCRIPT)
    def on_partial(msg):
        nonlocal current_partial, prev_full
        current_partial = TranscriptResult.from_message(msg).metadata.transcript or ""
        full = get_full_text()
        if full and full != prev_full:
            prev_full = full
            update_remaining(full)
            if not closed:
                loop = asyncio.get_event_loop()
                loop.create_task(ws.send_json({"type": "partial", "text": full}))

    try:
        await client.start_session(
            transcription_config=TranscriptionConfig(
                language="en",
                enable_partials=True,
                operating_point=OperatingPoint.ENHANCED,
            ),
            audio_format=AudioFormat(encoding="pcm_s16le", chunk_size=4096, sample_rate=16000),
        )
        await ws.send_json({"type": "session_started", "data": {"status": "connected"}})

        while True:
            message = await ws.receive()
            if "text" not in message:
                continue
            data = json.loads(message["text"])
            if data.get("type") == "audio_chunk":
                audio = base64.b64decode(data.get("audio_base_64", ""))
                await client.send_audio(audio)
            elif data.get("type") == "end_stream":
                print(f"🛑 Stream ended")
                print(f"   EN confirmed: {confirmed_text}")
                print(f"   KR confirmed: {translator.translated_confirmed}")
                print(f"   KR partial: {translator.translated_partial}")
                break

    except WebSocketDisconnect:
        print("👋 Disconnected")
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}")
    finally:
        closed = True
        await client.__aexit__(None, None, None)