"""
ElevenLabs Scribe v2 Speech-to-Text WebSocket handler.
Uses the same confirmed/partial translation pipeline as the Speechmatics router.
Endpoint: /stt/elevenlabs
"""

import os
import json
import time
import asyncio
import websockets

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from utils.tone import ToneDetector
from utils.speaker_pipeline import SpeakerPipeline
from auth.config import AUTH_ENABLED
from auth.dependencies import require_ws_auth

router = APIRouter()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id=scribe_v2_realtime"
CONNECTION_TIMEOUT = 10.0


@router.websocket("")
async def stream(ws: WebSocket):
    await ws.accept()
    user = await require_ws_auth(ws)
    if AUTH_ENABLED and user is None:
        return

    if not ELEVENLABS_API_KEY:
        await ws.send_json({"type": "error", "message": "ELEVENLABS_API_KEY not configured"})
        await ws.close()
        return

    closed = False
    stream_ending = False
    target_lang = ws.query_params.get("target_lang", "Korean")
    aggressiveness = int(ws.query_params.get("aggressiveness", "1"))
    confirm_punct_count = 1 if aggressiveness <= 1 else 2
    use_splitter = aggressiveness <= 1
    partial_interval = int(ws.query_params.get("update_frequency", "2"))
    translator_type = ws.query_params.get("translator", "realtime")
    use_realtime = translator_type == "realtime"
    use_deepl = translator_type == "deepl"
    async def on_tone_detected(tone: str):
        if not closed:
            await ws.send_json({"type": "tone_detected", "tone": tone})

    tone_detector = ToneDetector(target_lang=target_lang, on_detected=on_tone_detected)

    committed_text = ""
    current_partial = ""
    prev_partial = ""

    stream_start = time.time()
    speaker_id = "default"

    async def on_confirmed(text, elapsed_ms=0):
        if not closed:
            await ws.send_json({"type": "confirmed_translation", "speaker": speaker_id, "text": text, "elapsed_ms": elapsed_ms})

    async def on_partial(text, elapsed_ms=0):
        if not closed:
            await ws.send_json({"type": "partial_translation", "speaker": speaker_id, "text": text, "elapsed_ms": elapsed_ms})

    async def on_confirmed_transcript(text, elapsed_ms=0):
        if not closed:
            await ws.send_json({"type": "confirmed_transcript", "speaker": speaker_id, "text": text, "elapsed_ms": elapsed_ms})

    async def on_partial_transcript(text, elapsed_ms=0):
        if not closed:
            await ws.send_json({"type": "partial_transcript", "speaker": speaker_id, "text": text, "elapsed_ms": elapsed_ms})

    async def on_partial_delta(delta, generation, elapsed_ms=0):
        if not closed:
            await ws.send_json({"type": "partial_translation_delta", "speaker": speaker_id, "delta": delta, "generation": generation, "elapsed_ms": elapsed_ms})

    pipeline = SpeakerPipeline(
        speaker_id=speaker_id,
        on_confirmed=on_confirmed,
        on_partial=on_partial,
        on_confirmed_transcript=on_confirmed_transcript,
        on_partial_transcript=on_partial_transcript,
        target_lang=target_lang,
        tone_detector=tone_detector,
        stream_start=stream_start,
        confirm_punct_count=confirm_punct_count,
        use_splitter=use_splitter,
        partial_interval=partial_interval,
        use_realtime=use_realtime,
        use_deepl=use_deepl,
        on_partial_delta=on_partial_delta,
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
                    print("Stream ending, sending final commit")
                    stream_ending = True
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
        nonlocal committed_text, current_partial, prev_partial, closed

        try:
            async for message in elevenlabs_ws:
                data = json.loads(message)
                msg_type = data.get("message_type", "")
                text = data.get("text", "").strip()

                if msg_type == "committed_transcript":
                    if text:
                        committed_text = (committed_text + " " + text).strip() if committed_text else text
                        if committed_text != prev_partial:
                            prev_partial = committed_text
                            pipeline.feed(committed_text)

                    if stream_ending:
                        print("Stream ended")
                        print(f"   Source confirmed: {pipeline.prev_text}")
                        print(f"   Translated confirmed: {pipeline.translator.translated_confirmed}")
                        print(f"   Translated partial: {pipeline.translator.translated_partial}")
                        break

                elif msg_type == "partial_transcript":
                    current_partial = (committed_text + " " + text).strip() if committed_text else text
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
        await ws.send_json({"type": "tone_detecting"})

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


@router.websocket("/transcript")
async def stream_transcript_only(ws: WebSocket):
    """Transcription only — no translation pipeline."""
    await ws.accept()
    user = await require_ws_auth(ws)
    if AUTH_ENABLED and user is None:
        return

    if not ELEVENLABS_API_KEY:
        await ws.send_json({"type": "error", "message": "ELEVENLABS_API_KEY not configured"})
        await ws.close()
        return

    closed = False
    prev_partial = ""
    committed_text = ""  # all committed segments; used to restore context after EL resets its partial

    def trim_recent(text: str) -> str:
        """
        Keep at most 3 completed sentences + the current in-progress words.
        A sentence boundary is any .?! followed by a space.
        When more than 3 boundaries exist, drop everything up to and including
        the 4th-from-last boundary so exactly 3 remain.
        """
        boundaries = [i for i in range(len(text) - 1) if text[i] in ".?!" and text[i + 1] == " "]
        if len(boundaries) > 3:
            cut = boundaries[-4] + 2   # start of the 3rd-from-last sentence
            return text[cut:]
        return text

    async def forward_audio(elevenlabs_ws):
        try:
            while True:
                message = await ws.receive()
                if "text" not in message:
                    continue
                data = json.loads(message["text"])
                if data.get("type") == "audio_chunk":
                    await elevenlabs_ws.send(json.dumps({
                        "message_type": "input_audio_chunk",
                        "audio_base_64": data.get("audio_base_64", ""),
                        "commit": False,
                        "sample_rate": 16000,
                    }))
                elif data.get("type") == "end_stream":
                    await elevenlabs_ws.send(json.dumps({
                        "message_type": "input_audio_chunk",
                        "audio_base_64": "",
                        "commit": True,
                        "sample_rate": 16000,
                    }))
                    break
        except WebSocketDisconnect:
            pass

    async def forward_transcripts(elevenlabs_ws):
        nonlocal committed_text, prev_partial, closed
        try:
            async for message in elevenlabs_ws:
                data = json.loads(message)
                msg_type = data.get("message_type", "")
                text = data.get("text", "").strip()

                if msg_type == "committed_transcript":
                    committed_text = (committed_text + " " + text).strip() if committed_text else text
                    print(f"[EL committed] {text}")
                elif msg_type == "partial_transcript":
                    full = trim_recent((committed_text + " " + text).strip() if committed_text else text)
                    if full and full != prev_partial:
                        prev_partial = full
                        if not closed:
                            print(f"[EL partial  ] {full}")
                            await ws.send_json({"type": "partial_transcript", "speaker": "default", "text": full})
        except websockets.ConnectionClosed:
            pass

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
        await asyncio.gather(forward_audio(elevenlabs_ws), forward_transcripts(elevenlabs_ws))

    except asyncio.TimeoutError:
        await ws.send_json({"type": "error", "message": "ElevenLabs connection timeout"})
    except websockets.InvalidStatusCode as e:
        await ws.send_json({"type": "error", "message": str(e)})
    except WebSocketDisconnect:
        print("Disconnected (transcript-only)")
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
    finally:
        closed = True
        if elevenlabs_ws:
            await elevenlabs_ws.close()
