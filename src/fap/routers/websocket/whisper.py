# routers/websocket/whisper.py
"""
WebSocket route for Whisper ASR.

Direct integration with WhisperASR engine (no adapter layer).
Uses incremental mode with word-level timestamps.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from fap.asr.whisper_engine import WhisperASR
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


@router.websocket("/whisper")
async def whisper_stream(ws: WebSocket):
    """
    WebSocket endpoint for Whisper ASR.

    Direct integration with WhisperASR engine for lower latency.
    Uses incremental mode - words are finalized based on timestamps.
    """
    await ws.accept()
    print("WebSocket /ws/whisper connected")

    # Get shared model from app state
    shared_model = getattr(ws.app.state, "asr_model", None)

    # Create WhisperASR engine directly (no adapter)
    engine = WhisperASR(model=shared_model)
    print(f"Using WhisperASR engine directly")

    # SegmentManager in incremental mode for word-level timestamps
    segment_manager = SegmentManager(asr_mode="incremental")
    global_accumulator = GlobalAccumulator()

    pending_metadata = None
    current_segment_id = None

    try:
        while True:
            msg = await parse_websocket_message(ws)

            if msg["type"] == "end_stream":
                print("Client requested stream end")
                # Whisper doesn't have a finalize method that returns hypothesis
                # Just finalize the segment manager
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

                # Feed directly to WhisperASR engine
                hypothesis = engine.feed(audio)

                if hypothesis:
                    # Detect segment boundary
                    is_new_segment = hypothesis["segment_id"] != current_segment_id

                    if is_new_segment and current_segment_id is not None:
                        segment_manager = await handle_segment_boundary(
                            ws,
                            segment_manager,
                            global_accumulator,
                            asr_mode="incremental",
                        )
                        print(f"New segment: {hypothesis['segment_id']}")

                    current_segment_id = hypothesis["segment_id"]

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
        print("WebSocket /ws/whisper closed")
