import os
import json
import base64
import time
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from speechmatics.rt import (
    AsyncClient,
    ServerMessageType,
    TranscriptionConfig,
    TranscriptResult,
    OperatingPoint,
    AudioFormat,
)
from utils.tone import ToneDetector
from utils.speaker_pipeline import SpeakerPipeline

router = APIRouter()

SPEECHMATICS_API_KEY = os.getenv("SPEECHMATICS_API_KEY")


@router.websocket("")
async def stream(ws: WebSocket):
    await ws.accept()

    closed = False
    target_lang = ws.query_params.get("target_lang", "Korean")
    source_lang = ws.query_params.get("source_lang", "en")
    aggressiveness = int(ws.query_params.get("aggressiveness", "1"))
    confirm_punct_count = 1 if aggressiveness <= 1 else 2
    use_splitter = aggressiveness <= 1
    partial_interval = int(ws.query_params.get("update_frequency", "2"))
    use_realtime = True
    tone_detector = ToneDetector(target_lang=target_lang)

    stream_start = time.time()

    # Per-speaker state
    speaker_accumulated: dict[str, str] = {}
    pipelines: dict[str, SpeakerPipeline] = {}

    def get_or_create_pipeline(speaker_id: str) -> SpeakerPipeline:
        if speaker_id not in pipelines:
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

            pipelines[speaker_id] = SpeakerPipeline(
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
            )
        return pipelines[speaker_id]

    # ---- Speaker text parsing ----

    def parse_speaker_texts(results):
        """Parse per-speaker text from Speechmatics results."""
        texts: dict[str, str] = {}
        for r in results:
            if r.get("type") not in ("word", "punctuation"):
                continue
            content = r["alternatives"][0]["content"]
            speaker = r["alternatives"][0].get("speaker", "unknown")
            if speaker not in texts:
                texts[speaker] = ""
            if r["type"] == "punctuation":
                texts[speaker] = texts[speaker].rstrip() + content
            else:
                texts[speaker] += (" " + content) if texts[speaker] else content
        return texts

    def print_speaker_texts(partial_results=None):
        """Print accumulated + current partial per speaker."""
        merged = {s: t for s, t in speaker_accumulated.items()}
        if partial_results:
            for speaker, text in parse_speaker_texts(partial_results).items():
                merged[speaker] = (merged.get(speaker, "") + " " + text).strip()
        for speaker in sorted(merged):
            print(f"  🎤 {speaker}: {merged[speaker]}")

    def get_speaker_full_texts(partial_results=None):
        """Get accumulated + partial text per speaker for feeding into pipelines."""
        merged = {s: t for s, t in speaker_accumulated.items()}
        if partial_results:
            for speaker, text in parse_speaker_texts(partial_results).items():
                merged[speaker] = (merged.get(speaker, "") + " " + text).strip()
        return merged

    # ---- Speechmatics client ----

    client = AsyncClient(api_key=SPEECHMATICS_API_KEY)
    await client.__aenter__()

    @client.on(ServerMessageType.ADD_TRANSCRIPT)
    def on_final(msg):
        results = msg.get("results", [])
        parsed = parse_speaker_texts(results)
        for speaker, text in parsed.items():
            speaker_accumulated[speaker] = (speaker_accumulated.get(speaker, "") + " " + text).strip()

        # Feed each speaker's full text into their pipeline
        for speaker, full_text in speaker_accumulated.items():
            get_or_create_pipeline(speaker).feed(full_text)

        # print_speaker_texts()

    @client.on(ServerMessageType.ADD_PARTIAL_TRANSCRIPT)
    def on_partial(msg):
        results = msg.get("results", [])
        full_texts = get_speaker_full_texts(results)
        # print_speaker_texts(results)

        # Feed each speaker's accumulated + partial text into their pipeline
        for speaker, full_text in full_texts.items():
            get_or_create_pipeline(speaker).feed(full_text)

        # Send raw partial transcript to frontend
        transcript = TranscriptResult.from_message(msg).metadata.transcript or ""
        if transcript and not closed:
            loop = asyncio.get_event_loop()
            loop.create_task(ws.send_json({"type": "partial", "text": transcript}))

    try:
        await client.start_session(
            transcription_config=TranscriptionConfig(
                language=source_lang,
                enable_partials=True,
                operating_point=OperatingPoint.ENHANCED,
                diarization="speaker",
                speaker_diarization_config={
                    "max_speakers": 10
                }
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
                print("🛑 Stream ended")
                for speaker_id, pipeline in sorted(pipelines.items()):
                    print(f"  🎤 {speaker_id}: confirmed={pipeline.translator.translated_confirmed}")
                break

    except WebSocketDisconnect:
        print("👋 Disconnected")
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}")
    finally:
        closed = True
        await client.__aexit__(None, None, None)
