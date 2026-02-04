"""
ElevenLabs Scribe v2 Speech-to-Text WebSocket handler (Experimental v2)
Endpoint: /stt/elevenlabs-two

Uses StreamingTranslationWorker for full transcript streaming experiments.
"""

import os
import json
import asyncio
import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from utils.translationTwo import StreamingTranslationWorker

router = APIRouter()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id=scribe_v2_realtime"
CONNECTION_TIMEOUT = 10.0


@router.websocket("")
async def stream(ws: WebSocket):
    """Experimental WebSocket endpoint for ElevenLabs STT + Streaming Translation"""
    await ws.accept()
    print("\n" + "="*60)
    print("SESSION START (EXPERIMENTAL V2)")
    print("="*60)

    if not ELEVENLABS_API_KEY:
        await ws.send_json({"type": "error", "message": "ELEVENLABS_API_KEY not configured"})
        await ws.close()
        return

    elevenlabs_ws = None
    translator = StreamingTranslationWorker(ws)
    translator.start()

    try:
        # Connect to ElevenLabs
        print("Connecting to ElevenLabs...")
        elevenlabs_ws = await asyncio.wait_for(
            websockets.connect(
                ELEVENLABS_URL,
                additional_headers={"xi-api-key": ELEVENLABS_API_KEY},
            ),
            timeout=CONNECTION_TIMEOUT
        )

        # Get session
        session_msg = await elevenlabs_ws.recv()
        session_data = json.loads(session_msg)
        print(f"Connected. Session: {session_data.get('session_id')[:8]}...")
        print("-"*60)

        await ws.send_json({"type": "session_started", "data": session_data})

        # Run audio forwarding and transcript processing
        await asyncio.gather(
            forward_audio(ws, elevenlabs_ws, translator),
            forward_transcripts(ws, elevenlabs_ws, translator),
        )

    except asyncio.TimeoutError:
        print("ERROR: Connection timeout")
        await ws.send_json({"type": "error", "message": "Connection timeout"})
    except websockets.InvalidStatusCode as e:
        print(f"ERROR: ElevenLabs connection failed: {e}")
        await ws.send_json({"type": "error", "message": str(e)})
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
    finally:
        if elevenlabs_ws:
            await elevenlabs_ws.close()
        
        await asyncio.sleep(0.3)
        await translator.shutdown()
        
        print("-"*60)
        print("SESSION END")
        print("="*60 + "\n")


async def forward_audio(client_ws: WebSocket, elevenlabs_ws, translator: StreamingTranslationWorker):
    """Forward audio chunks from client to ElevenLabs"""
    chunk_count = 0
    
    try:
        while not translator.shutdown_event.is_set():
            message = await client_ws.receive()

            if "text" not in message:
                continue

            data = json.loads(message["text"])
            msg_type = data.get("type")

            if msg_type == "audio_chunk":
                chunk_count += 1
                await elevenlabs_ws.send(json.dumps({
                    "message_type": "input_audio_chunk",
                    "audio_base_64": data.get("audio_base_64", ""),
                    "commit": False,
                    "sample_rate": 16000,
                }))

            elif msg_type == "end_stream":
                print(f"\nStream ended ({chunk_count} chunks)")
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
        print(f"ERROR in forward_audio: {type(e).__name__}: {e}")
    finally:
        translator.shutdown_event.set()


async def forward_transcripts(client_ws: WebSocket, elevenlabs_ws, translator: StreamingTranslationWorker):
    """Forward transcripts from ElevenLabs to client and process for translation"""
    
    try:
        async for message in elevenlabs_ws:
            if translator.shutdown_event.is_set():
                break

            data = json.loads(message)
            msg_type = data.get("message_type", "unknown")
            text = data.get("text", "").strip()

            # Forward to client (for frontend display)
            await client_ws.send_json({"type": msg_type, "data": data})

            # Process for translation (log everything)
            if text and msg_type in ["partial_transcript", "committed_transcript"]:
                is_committed = (msg_type == "committed_transcript")
                await translator.process_transcript(text, is_committed)

    except websockets.ConnectionClosed:
        pass
    except Exception as e:
        print(f"ERROR in forward_transcripts: {type(e).__name__}: {e}")
    finally:
        translator.shutdown_event.set()