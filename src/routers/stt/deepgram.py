"""
Deepgram Nova-3 Streaming Speech-to-Text WebSocket handler.
Endpoint: /stt/deepgram
"""

import os
import json
import asyncio
import base64
import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from urllib.parse import urlencode

router = APIRouter()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_BASE_URL = "wss://api.deepgram.com/v1/listen"
CONNECTION_TIMEOUT = 10.0


@router.websocket("")
async def stream(ws: WebSocket):
    await ws.accept()

    if not DEEPGRAM_API_KEY:
        await ws.send_json({"type": "error", "message": "DEEPGRAM_API_KEY not configured"})
        await ws.close()
        return

    closed = False
    # source_lang = ws.query_params.get("source_lang", "ko")
    source_lang = "ko"

    params = urlencode({
        "model": "nova-3",
        "language": source_lang,
        "punctuate": "true",
        "smart_format": "true",
        "interim_results": "true",
        "endpointing": 10,
        "diarize": "true",
        "encoding": "linear16",
        "sample_rate": 16000,
        "channels": 1,
    })
    deepgram_url = f"{DEEPGRAM_BASE_URL}?{params}"

    async def forward_audio(dg_ws):
        try:
            while True:
                message = await ws.receive()
                if "text" not in message:
                    continue

                data = json.loads(message["text"])
                msg_type = data.get("type")

                if msg_type == "audio_chunk":
                    audio = base64.b64decode(data.get("audio_base_64", ""))
                    await dg_ws.send(audio)
                elif msg_type == "end_stream":
                    print("Stream ending, sending CloseStream")
                    await dg_ws.send(json.dumps({"type": "CloseStream"}))
                    break

        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"forward_audio error: {type(e).__name__}: {e}")

    async def forward_transcripts(dg_ws):
        nonlocal closed
        try:
            async for message in dg_ws:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "Results":
                    channel = data.get("channel", {})
                    alternatives = channel.get("alternatives", [])
                    if not alternatives:
                        continue

                    transcript = alternatives[0].get("transcript", "")
                    is_final = data.get("is_final", False)
                    speech_final = data.get("speech_final", False)

                    if not transcript:
                        continue

                    # Extract speaker info from words
                    words = alternatives[0].get("words", [])
                    speakers = set(w.get("speaker") for w in words if "speaker" in w)
                    speaker = None
                    if words and "speaker" in words[0]:
                        speaker = f"{words[0]['speaker']}+" if len(speakers) > 1 else words[0]["speaker"]

                    prefix = "🟢 Final" if is_final else "⏳ Interim"
                    speaker_str = f" [Speaker {speaker}]" if speaker is not None else ""
                    print(f"{prefix}{speaker_str}: \"{transcript}\" (speech_final={speech_final})")

                    if not closed:
                        await ws.send_json({"type": "partial", "text": transcript})

                elif msg_type == "Metadata":
                    request_id = data.get("request_id", "")
                    print(f"Deepgram session: {request_id}")
                    if not closed:
                        await ws.send_json({"type": "session_started", "data": {"status": "connected", "request_id": request_id}})

        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            print(f"forward_transcripts error: {type(e).__name__}: {e}")

    dg_ws = None

    try:
        dg_ws = await asyncio.wait_for(
            websockets.connect(
                deepgram_url,
                additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
            ),
            timeout=CONNECTION_TIMEOUT,
        )

        await asyncio.gather(
            forward_audio(dg_ws),
            forward_transcripts(dg_ws),
        )

    except asyncio.TimeoutError:
        await ws.send_json({"type": "error", "message": "Deepgram connection timeout"})
    except websockets.InvalidStatusCode as e:
        print(f"Deepgram connection failed: {e}")
        await ws.send_json({"type": "error", "message": str(e)})
    except WebSocketDisconnect:
        print("Disconnected")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
    finally:
        closed = True
        if dg_ws:
            await dg_ws.close()
