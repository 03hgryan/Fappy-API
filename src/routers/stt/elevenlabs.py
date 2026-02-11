"""
ElevenLabs Scribe v2 Speech-to-Text WebSocket handler.
Uses the same confirmed/partial translation pipeline as the Speechmatics router.
Endpoint: /stt/elevenlabs
"""

import os
import json
import re
import asyncio
import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from utils.translation import Translator
from utils.tone import ToneDetector

router = APIRouter()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id=scribe_v2_realtime"
CONNECTION_TIMEOUT = 10.0


@router.websocket("")
async def stream(ws: WebSocket):
    await ws.accept()

    if not ELEVENLABS_API_KEY:
        await ws.send_json({"type": "error", "message": "ELEVENLABS_API_KEY not configured"})
        await ws.close()
        return

    committed_text = ""
    current_partial = ""
    prev_full = ""
    confirmed_text = ""
    confirmed_word_count = 0
    prev_remaining = ""
    partial_count = 0
    PARTIAL_INTERVAL = 2
    CONFIRM_PUNCT_COUNT = 1
    closed = False

    async def on_confirmed_translation(confirmed_korean):
        if not closed:
            await ws.send_json({"type": "confirmed_translation", "text": confirmed_korean})

    async def on_partial_translation(partial_korean):
        if not closed:
            await ws.send_json({"type": "partial_translation", "text": partial_korean})

    tone_detector = ToneDetector()
    translator = Translator(
        on_confirmed=on_confirmed_translation,
        on_partial=on_partial_translation,
        tone_detector=tone_detector,
    )

    def get_full_text():
        if not committed_text:
            return current_partial.strip()
        if not current_partial:
            return committed_text.strip()
        f = committed_text.rstrip()
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

        matches = list(re.finditer(r'[.?!]\s+\w', remaining_text))
        if len(matches) >= CONFIRM_PUNCT_COUNT:
            cut_match = matches[-CONFIRM_PUNCT_COUNT]
            cut = cut_match.start() + 1
            new_confirmed = remaining_text[:cut].strip()

            if new_confirmed:
                confirmed_text = (confirmed_text + " " + new_confirmed).strip() if confirmed_text else new_confirmed
                confirmed_word_count += len(new_confirmed.split())
                remaining_text = " ".join(words[confirmed_word_count:])
                print(f'confirmed: "{new_confirmed}"')
                loop = asyncio.get_event_loop()
                loop.create_task(translator.translate_confirmed(new_confirmed))
                partial_count = 0

        partial_count += 1
        if partial_count % PARTIAL_INTERVAL == 0 and remaining_text:
            loop = asyncio.get_event_loop()
            loop.create_task(translator.translate_partial(remaining_text))

        if remaining_text != prev_remaining:
            prev_remaining = remaining_text

    async def forward_audio(elevenlabs_ws):
        try:
            while True:
                message = await ws.receive()
                if "text" not in message:
                    continue

                data = json.loads(message["text"])
                msg_type = data.get("type")

                if msg_type == "audio_chunk":
                    await elevenlabs_ws.send(json.dumps({
                        "message_type": "input_audio_chunk",
                        "audio_base_64": data.get("audio_base_64", ""),
                        "commit": False,
                        "sample_rate": 16000,
                    }))
                elif msg_type == "end_stream":
                    print("Stream ended")
                    await elevenlabs_ws.send(json.dumps({
                        "message_type": "input_audio_chunk",
                        "audio_base_64": "",
                        "commit": True,
                        "sample_rate": 16000,
                    }))
                    break

        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"forward_audio error: {type(e).__name__}: {e}")

    async def forward_transcripts(elevenlabs_ws):
        """Forward transcripts from ElevenLabs to client and process for translation"""
        nonlocal committed_text, current_partial, prev_full, closed

        try:
            async for message in elevenlabs_ws:
                data = json.loads(message)
                msg_type = data.get("message_type", "")
                text = data.get("text", "").strip()

                if msg_type == "committed_transcript":
                    if text:
                        committed_text = committed_text + text if committed_text else text
                        full = get_full_text()
                        if full != prev_full:
                            prev_full = full
                            update_remaining(full)

                    print(f"Stream ended")
                    print(f"   EN confirmed: {confirmed_text}")
                    print(f"   KR confirmed: {translator.translated_confirmed}")
                    print(f"   KR partial: {translator.translated_partial}")
                    break

                elif msg_type == "partial_transcript":
                    current_partial = text
                    full = get_full_text()
                    if full and full != prev_full:
                        prev_full = full
                        update_remaining(full)
                        if not closed:
                            await ws.send_json({"type": "partial", "text": full})

        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            print(f"forward_transcripts error: {type(e).__name__}: {e}")

    elevenlabs_ws = None

    try:
        elevenlabs_ws = await asyncio.wait_for(
            websockets.connect(
                ELEVENLABS_URL,
                additional_headers={"xi-api-key": ELEVENLABS_API_KEY},
            ),
            timeout=CONNECTION_TIMEOUT,
        )

        session_msg = await elevenlabs_ws.recv()
        session_data = json.loads(session_msg)
        await ws.send_json({"type": "session_started", "data": session_data})

        await asyncio.gather(
            forward_audio(elevenlabs_ws),
            forward_transcripts(elevenlabs_ws),
        )

    except asyncio.TimeoutError:
        await ws.send_json({"type": "error", "message": "ElevenLabs connection timeout"})
    except websockets.InvalidStatusCode as e:
        print(f"ElevenLabs connection failed: {e}")
        await ws.send_json({"type": "error", "message": str(e)})
    except WebSocketDisconnect:
        print("Disconnected")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
    finally:
        closed = True
        if elevenlabs_ws:
            await elevenlabs_ws.close()
