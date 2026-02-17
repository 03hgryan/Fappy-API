"""
AssemblyAI v3 Streaming Speech-to-Text WebSocket handler.
Endpoint: /stt/assemblyai
"""

import os
import json
import asyncio
import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from urllib.parse import urlencode

router = APIRouter()

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
ASSEMBLYAI_BASE_URL = "wss://streaming.assemblyai.com/v3/ws"
CONNECTION_TIMEOUT = 10.0


@router.websocket("")
async def stream(ws: WebSocket):
    await ws.accept()

    if not ASSEMBLYAI_API_KEY:
        await ws.send_json({"type": "error", "message": "ASSEMBLYAI_API_KEY not configured"})
        await ws.close()
        return

    closed = False
    stream_ending = False

    params = urlencode({"sample_rate": 16000, "format_turns": True})
    assemblyai_url = f"{ASSEMBLYAI_BASE_URL}?{params}"

    async def forward_audio(aai_ws):
        nonlocal stream_ending
        try:
            while True:
                message = await ws.receive()
                if "text" not in message:
                    continue

                data = json.loads(message["text"])
                msg_type = data.get("type")

                if msg_type == "audio_chunk":
                    import base64
                    audio = base64.b64decode(data.get("audio_base_64", ""))
                    await aai_ws.send(audio)
                elif msg_type == "end_stream":
                    print("Stream ending, sending terminate")
                    stream_ending = True
                    await aai_ws.send(json.dumps({"type": "Terminate"}))
                    break

        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"forward_audio error: {type(e).__name__}: {e}")

    async def forward_transcripts(aai_ws):
        nonlocal closed
        try:
            async for message in aai_ws:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "Begin":
                    session_id = data.get("id")
                    print(f"AssemblyAI session began: {session_id}")
                    await ws.send_json({"type": "session_started", "data": {"status": "connected", "session_id": session_id}})

                elif msg_type == "Turn":
                    transcript = data.get("transcript", "")
                    formatted = data.get("turn_is_formatted", False)

                    if formatted:
                        print(f"🟢 [AssemblyAI] Final: \"{transcript}\"")
                    else:
                        print(f"⏳ [AssemblyAI] Partial: \"{transcript}\"")

                    if not closed:
                        await ws.send_json({"type": "partial", "text": transcript})

                elif msg_type == "Termination":
                    audio_dur = data.get("audio_duration_seconds", 0)
                    session_dur = data.get("session_duration_seconds", 0)
                    print(f"AssemblyAI terminated: audio={audio_dur}s session={session_dur}s")
                    break

        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            print(f"forward_transcripts error: {type(e).__name__}: {e}")

    aai_ws = None

    try:
        aai_ws = await asyncio.wait_for(
            websockets.connect(
                assemblyai_url,
                additional_headers={"Authorization": ASSEMBLYAI_API_KEY},
            ),
            timeout=CONNECTION_TIMEOUT,
        )

        await asyncio.gather(
            forward_audio(aai_ws),
            forward_transcripts(aai_ws),
        )

    except asyncio.TimeoutError:
        await ws.send_json({"type": "error", "message": "AssemblyAI connection timeout"})
    except websockets.InvalidStatusCode as e:
        print(f"AssemblyAI connection failed: {e}")
        await ws.send_json({"type": "error", "message": str(e)})
    except WebSocketDisconnect:
        print("Disconnected")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
    finally:
        closed = True
        if aai_ws:
            await aai_ws.close()
