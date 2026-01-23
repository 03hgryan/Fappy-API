# routers/websocket/legacy.py
"""
Legacy WebSocket route for backwards compatibility.

Maintains /ws/stream endpoint that delegates to the appropriate
provider-specific handler based on ASR_PROVIDER environment variable.

DEPRECATED: Use /ws/whisper, /ws/google, or /ws/openai instead.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from fap.asr import create_asr_adapter
from fap.utils.segmentManager import SegmentManager
from fap.utils.globalAccumulator import GlobalAccumulator

from ._base import (
    parse_websocket_message,
    send_segment_update,
    send_segment_finalized,
    send_transcript_final,
    handle_segment_boundary,
    handle_finalized_words,
)

router = APIRouter()


@router.websocket("/stream")
async def legacy_stream(ws: WebSocket):
    """
    Legacy WebSocket endpoint for streaming ASR.

    DEPRECATED: This endpoint uses the adapter pattern which adds latency.
    Prefer using the direct endpoints instead:
    - /ws/whisper - For local Whisper ASR
    - /ws/google  - For Google Cloud Speech-to-Text
    - /ws/openai  - For OpenAI Realtime API

    This endpoint auto-detects the provider from ASR_PROVIDER env var.
    """
    await ws.accept()
    print("WebSocket /ws/stream connected (DEPRECATED - use /ws/{provider} instead)")

    # Send deprecation warning to client
    await ws.send_json({
        "type": "warning",
        "message": "This endpoint is deprecated. Use /ws/whisper, /ws/google, or /ws/openai for lower latency.",
    })

    # Get shared model and provider from app state
    shared_model = getattr(ws.app.state, "asr_model", None)
    provider = getattr(ws.app.state, "asr_provider", "whisper")

    # Create ASR adapter based on provider
    if provider == "whisper":
        asr_adapter = create_asr_adapter(
            provider="whisper",
            model=shared_model,
        )
    elif provider == "google":
        asr_adapter = create_asr_adapter(
            provider="google",
        )
    elif provider == "openai":
        asr_adapter = create_asr_adapter(
            provider="openai",
        )
    else:
        # Fallback to whisper
        asr_adapter = create_asr_adapter(
            provider="whisper",
            model=shared_model,
        )

    print(f"Using ASR adapter: {asr_adapter.provider_name}")

    # Create SegmentManager with appropriate mode
    asr_mode = "rewriting" if provider in ("google", "openai") else "incremental"
    segment_manager = SegmentManager(asr_mode=asr_mode)
    global_accumulator = GlobalAccumulator()

    pending_metadata = None
    current_segment_id = None

    try:
        while True:
            msg = await parse_websocket_message(ws)

            if msg["type"] == "end_stream":
                print("Client requested stream end")

                # Finalize adapter
                final_hypothesis = asr_adapter.finalize()
                if final_hypothesis:
                    segment_output = segment_manager.ingest(final_hypothesis)
                    if segment_output["finalized_words"]:
                        global_accumulator.append_words(
                            segment_output["segment_id"],
                            segment_output["finalized_words"]
                        )

                # Finalize active segment
                final_output = segment_manager.finalize()

                if final_output:
                    if final_output["finalized_words"]:
                        global_accumulator.append_words(
                            final_output["segment_id"],
                            final_output["finalized_words"]
                        )

                    await send_segment_finalized(ws, final_output)
                    print(f"Final transcript: \"{final_output['rendered_text']['stable']}\"")

                # Send full accumulated transcript
                await send_transcript_final(ws, global_accumulator)

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

                # Feed to ASR adapter
                hypothesis = asr_adapter.feed(audio)

                if hypothesis:
                    # Detect segment boundary
                    is_new_segment = hypothesis["segment_id"] != current_segment_id

                    if is_new_segment and current_segment_id is not None:
                        segment_manager = await handle_segment_boundary(
                            ws,
                            segment_manager,
                            global_accumulator,
                            asr_mode=asr_mode,
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
        if asr_adapter.is_streaming:
            asr_adapter.reset()
        print("WebSocket /ws/stream closed")
