# routers/websocket/google.py
"""
WebSocket route for Google Cloud ASR.

Direct integration with GoogleCloudASR engine (no adapter layer).
Uses rewriting mode with is_final detection.
"""

import os
import queue

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from fap.asr.google_engine import GoogleCloudASR
from fap.utils.segmentManager import SegmentManager
from fap.utils.globalAccumulator import GlobalAccumulator

from ._base import (
    parse_websocket_message,
    send_segment_update,
    handle_segment_boundary,
    handle_finalized_words,
    handle_stream_end,
)

router = APIRouter()


def drain_finals_queue(engine: GoogleCloudASR) -> list[dict]:
    """Drain all finals from engine's finals queue."""
    finals = []
    while True:
        try:
            final = engine._finals_queue.get_nowait()
            if final:
                finals.append(final)
        except queue.Empty:
            break
    return finals


def get_latest_interim(engine: GoogleCloudASR) -> dict | None:
    """Get the latest interim result from the response queue."""
    latest = None
    while True:
        try:
            result = engine._response_queue.get_nowait()
            if result and not result.get("is_final_from_google", False):
                latest = result
        except queue.Empty:
            break
    return latest


@router.websocket("/google")
async def google_stream(ws: WebSocket):
    """
    WebSocket endpoint for Google Cloud ASR.

    Direct integration with GoogleCloudASR engine for lower latency.
    Uses rewriting mode - handles is_final_from_google natively.
    """
    await ws.accept()
    print("WebSocket /ws/google connected")

    # Create GoogleCloudASR engine directly (no adapter)
    engine = GoogleCloudASR(
        language_code=os.getenv("GOOGLE_ASR_LANGUAGE", "en-US"),
        credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        model=os.getenv("GOOGLE_ASR_MODEL", "latest_long"),
    )
    print(f"Using GoogleCloudASR engine directly")

    # SegmentManager in rewriting mode for streaming APIs
    segment_manager = SegmentManager(asr_mode="rewriting")
    global_accumulator = GlobalAccumulator()

    pending_metadata = None
    current_segment_id = None
    revision = 0

    try:
        while True:
            msg = await parse_websocket_message(ws)

            if msg["type"] == "end_stream":
                print("Client requested stream end")

                # Brief wait for any pending finals
                import asyncio
                await asyncio.sleep(0.1)

                # Drain any remaining finals
                finals = drain_finals_queue(engine)
                for final in finals:
                    hypothesis = _normalize_hypothesis(final, revision, is_final=True)
                    revision += 1
                    segment_output = segment_manager.ingest(hypothesis)
                    if segment_output["finalized_words"]:
                        global_accumulator.append_words(
                            segment_output["segment_id"],
                            segment_output["finalized_words"]
                        )

                await handle_stream_end(
                    ws,
                    segment_manager,
                    global_accumulator,
                    final_hypothesis=None,
                )
                await ws.close()
                return

            elif msg["type"] == "metadata":
                pending_metadata = msg["data"]
                print(f"Metadata: {pending_metadata}")

            elif msg["type"] == "audio":
                if pending_metadata is None:
                    print("Received audio without metadata, skipping")
                    continue

                audio = msg["data"]
                print(f"Audio: {len(audio)} bytes, chunk #{pending_metadata.get('chunk_index')}")

                # Check for finals that arrived before feeding
                finals = drain_finals_queue(engine)
                for final in finals:
                    hypothesis = _normalize_hypothesis(final, revision, is_final=True)
                    revision += 1
                    await _process_hypothesis(
                        ws, hypothesis, segment_manager, global_accumulator,
                        current_segment_id
                    )
                    current_segment_id = hypothesis["segment_id"]

                # Feed audio directly to engine
                engine.feed(audio)

                # Small delay then check for results (faster than adapter's 20ms)
                import asyncio
                await asyncio.sleep(0.01)  # 10ms

                # Check for finals again
                finals = drain_finals_queue(engine)
                for final in finals:
                    hypothesis = _normalize_hypothesis(final, revision, is_final=True)
                    revision += 1
                    result = await _process_hypothesis(
                        ws, hypothesis, segment_manager, global_accumulator,
                        current_segment_id
                    )
                    segment_manager = result["segment_manager"]
                    current_segment_id = hypothesis["segment_id"]

                # Get latest interim
                interim = get_latest_interim(engine)
                if interim:
                    hypothesis = _normalize_hypothesis(interim, revision, is_final=False)
                    revision += 1
                    result = await _process_hypothesis(
                        ws, hypothesis, segment_manager, global_accumulator,
                        current_segment_id
                    )
                    segment_manager = result["segment_manager"]
                    current_segment_id = hypothesis["segment_id"]

                pending_metadata = None

    except WebSocketDisconnect:
        print("Client disconnected: WebSocketDisconnect")
    except RuntimeError as e:
        if "disconnect message has been received" in str(e):
            print("Client disconnected: Runtime")
        else:
            print(f"WebSocket error: RuntimeError: {e}")
    except Exception as e:
        print(f"WebSocket error: {type(e).__name__}: {e}")
    finally:
        engine.reset()
        print("WebSocket /ws/google closed")


def _normalize_hypothesis(hypothesis: dict, revision: int, is_final: bool) -> dict:
    """Normalize Google hypothesis to standard format."""
    import time
    return {
        "type": "hypothesis",
        "segment_id": hypothesis.get("segment_id", f"google-{int(time.time() * 1000)}"),
        "revision": revision,
        "text": hypothesis.get("text", ""),
        "words": hypothesis.get("words", []),
        "confidence": hypothesis.get("confidence", 0.9),
        "start_time_ms": hypothesis.get("start_time_ms", 0),
        "end_time_ms": hypothesis.get("end_time_ms", 0),
        "is_final": is_final,
    }


async def _process_hypothesis(
    ws: WebSocket,
    hypothesis: dict,
    segment_manager: SegmentManager,
    global_accumulator: GlobalAccumulator,
    current_segment_id: str | None,
) -> dict:
    """
    Process a hypothesis through the segment manager and send updates.

    Returns:
        dict with updated segment_manager
    """
    # Detect segment boundary
    is_new_segment = hypothesis["segment_id"] != current_segment_id

    if is_new_segment and current_segment_id is not None:
        segment_manager = await handle_segment_boundary(
            ws,
            segment_manager,
            global_accumulator,
            asr_mode="rewriting",
        )
        print(f"New segment: {hypothesis['segment_id']}")

    # Process through SegmentManager
    segment_output = segment_manager.ingest(hypothesis)

    # Handle finalized words
    await handle_finalized_words(segment_output, global_accumulator)

    # Send live update to client
    await send_segment_update(ws, segment_output)

    # Logging
    stable = segment_output["rendered_text"]["stable"]
    unstable = segment_output["rendered_text"]["unstable"]
    print(f"Stable:   \"{stable}\"")
    print(f"Unstable: \"{unstable}\"")
    print(f"Revision {segment_output['revision']}")

    return {"segment_manager": segment_manager}
