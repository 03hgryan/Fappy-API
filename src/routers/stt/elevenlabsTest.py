import os
import json
import asyncio
import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from utils.translationTest import Translator
from utils.combiner import Combiner

router = APIRouter()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id=scribe_v2_realtime"
CONNECTION_TIMEOUT = 10.0


@router.websocket("")
async def stream(ws: WebSocket):
    await ws.accept()
    print("\n" + "=" * 60)
    print("SESSION START")
    print("=" * 60)

    if not ELEVENLABS_API_KEY:
        await ws.send_json({"type": "error", "message": "ELEVENLABS_API_KEY not configured"})
        await ws.close()
        return

    elevenlabs_ws = None

    async def on_combined(full_text, seq):
        await ws.send_json({
            "type": "combined",
            "seq": seq,
            "full": full_text,
        })

    combiner = Combiner(on_combined=on_combined)

    async def on_translation(english_source, translated_text, seq):
        await ws.send_json({"type": "translation", "seq": seq, "text": translated_text})
        combiner.feed_translation(english_source, translated_text)

    translator = Translator(on_translation=on_translation)

    try:
        print("Connecting to ElevenLabs...")
        elevenlabs_ws = await asyncio.wait_for(
            websockets.connect(
                ELEVENLABS_URL,
                additional_headers={"xi-api-key": ELEVENLABS_API_KEY},
            ),
            timeout=CONNECTION_TIMEOUT,
        )

        session_msg = await elevenlabs_ws.recv()
        session_data = json.loads(session_msg)
        print(f"Connected. Session: {session_data.get('session_id')[:8]}...")
        print("-" * 60)

        await ws.send_json({"type": "session_started", "data": session_data})

        await asyncio.gather(
            forward_audio(ws, elevenlabs_ws),
            forward_transcripts(ws, elevenlabs_ws, translator),
        )

    except asyncio.TimeoutError:
        await ws.send_json({"type": "error", "message": "Connection timeout"})
    except websockets.InvalidStatusCode as e:
        await ws.send_json({"type": "error", "message": str(e)})
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
    finally:
        if elevenlabs_ws:
            await elevenlabs_ws.close()
        print("=" * 60 + "\n")


async def forward_audio(client_ws: WebSocket, elevenlabs_ws):
    chunk_count = 0
    try:
        while True:
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
                print(f"Stream ended ({chunk_count} chunks)")
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


async def forward_transcripts(client_ws: WebSocket, elevenlabs_ws, translator: Translator):
    prev_text = ""
    silent = False
    committed_text = ""  # All committed transcripts concatenated

    try:
        async for message in elevenlabs_ws:
            data = json.loads(message)
            msg_type = data.get("message_type", "unknown")
            text = data.get("text", "").strip()

            if not text or msg_type not in ["partial_transcript", "committed_transcript"]:
                continue

            if msg_type == "committed_transcript":
                # Accumulate committed text
                if committed_text:
                    committed_text += " " + text
                else:
                    committed_text = text
                continue

            # For partials: build full running text = committed + current partial
            full_text = (committed_text + " " + text).strip() if committed_text else text

            if full_text == prev_text:
                if not silent:
                    silent = True
                    print(f"🔇 Silence")
                continue
            else:
                if silent:
                    silent = False
                    print(f"🔊 Speech resumed")
                prev_text = full_text

            await client_ws.send_json({"type": "partial", "text": full_text})
            print(f"📝 {full_text[-80:]}")

            translator.feed_partial(full_text)

    except websockets.ConnectionClosed:
        pass
    except Exception as e:
        print(f"ERROR in forward_transcripts: {type(e).__name__}: {e}")