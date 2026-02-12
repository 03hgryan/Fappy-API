"""
ElevenLabs Scribe v2 Speech-to-Text WebSocket handler.
Uses the same confirmed/partial translation pipeline as the Speechmatics router.
Endpoint: /stt/elevenlabs
"""

import os
import json
import asyncio
import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from utils.tone import ToneDetector
from utils.speaker_pipeline import SpeakerPipeline

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

    closed = False
    target_lang = ws.query_params.get("target_lang", "Korean")
    tone_detector = ToneDetector(target_lang=target_lang)

    current_partial = ""
    prev_partial = ""

    speaker_id = "default"

    async def on_confirmed(text):
        if not closed:
            await ws.send_json({"type": "confirmed_translation", "speaker": speaker_id, "text": text})

    async def on_partial(text):
        if not closed:
            await ws.send_json({"type": "partial_translation", "speaker": speaker_id, "text": text})

    pipeline = SpeakerPipeline(
        speaker_id=speaker_id,
        on_confirmed=on_confirmed,
        on_partial=on_partial,
        target_lang=target_lang,
        tone_detector=tone_detector,
    )

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
        nonlocal current_partial, prev_partial, closed

        try:
            async for message in elevenlabs_ws:
                data = json.loads(message)
                msg_type = data.get("message_type", "")
                text = data.get("text", "").strip()

                if msg_type == "committed_transcript":
                    if text:
                        current_partial = text
                        if current_partial != prev_partial:
                            prev_partial = current_partial
                            pipeline.feed(current_partial)

                    print("Stream ended")
                    print(f"   Source confirmed: {pipeline.prev_text}")
                    print(f"   Translated confirmed: {pipeline.translator.translated_confirmed}")
                    print(f"   Translated partial: {pipeline.translator.translated_partial}")
                    break

                elif msg_type == "partial_transcript":
                    current_partial = text
                    if current_partial and current_partial != prev_partial:
                        prev_partial = current_partial
                        pipeline.feed(current_partial)
                        if not closed:
                            await ws.send_json({"type": "partial", "text": current_partial})

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
