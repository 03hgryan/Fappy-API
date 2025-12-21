from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fap.asr import StreamingASR
from fap.utils.segmentManager import SegmentManager
from fap.utils.globalAccumulator import GlobalAccumulator

router = APIRouter()


# Audio decoder (passthrough for PCM16)
def decode_audio(msg: bytes) -> bytes:
    """Decode incoming audio bytes"""
    return msg  # Already PCM16 from frontend


# ASR is created per-connection for isolated state
# segment_manager is created per-connection for isolated state


@router.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    print("✅ WebSocket connection accepted")

    # Get shared model from app state
    shared_model = ws.app.state.asr_model

    # Each client gets isolated state (buffer, segment_id, etc.) but shares the model
    asr_instance = StreamingASR(model_size="base", device="cpu", model=shared_model)
    segment_manager_instance = SegmentManager()
    global_accumulator = GlobalAccumulator()  # Layer 3: Immutable history
    pending_metadata = None
    current_segment_id = None  # Track active segment for boundary detection
    current_segment_finalized = False  # Prevent duplicate finalization of current segment

    try:
        while True:
            # Receive message (handle both binary and text)
            message = await ws.receive()

            # Handle metadata (JSON)
            if "text" in message:
                import json

                data = json.loads(message["text"])

                # Handle end_stream signal from client
                if data.get("type") == "end_stream":
                    print("🏁 Client requested stream end")

                    # Finalize active segment (only if not already finalized)
                    if not current_segment_finalized:
                        final_segment = segment_manager_instance.finalize()
                        if final_segment and final_segment["committed"]:
                            print(f"🏁 Finalized: \"{final_segment['committed']}\"")

                            # Layer 3: Append to immutable accumulator
                            finalized = global_accumulator.append(
                                current_segment_id,
                                final_segment["committed"]
                            )

                            # Send immutable segment to client
                            await ws.send_json({
                                "type": "segments_finalized",
                                "segments": [finalized],
                            })

                            current_segment_finalized = True

                    # Close cleanly
                    await ws.close()
                    return

                # Regular metadata
                pending_metadata = data
                print(f"📝 Metadata: {pending_metadata}")

            # Handle audio (binary PCM)
            elif "bytes" in message:
                if pending_metadata is None:
                    print("⚠️ Received audio without metadata, skipping")
                    continue

                audio = decode_audio(message["bytes"])
                print(
                    f"🎵 Audio: {len(audio)} bytes, chunk #{pending_metadata.get('chunk_index')}"
                )

                # Feed to ASR (may return None if buffer not ready)
                hypothesis = asr_instance.feed(audio)

                if hypothesis:
                    # Detect segment boundary (silence-based from ASR)
                    is_new_segment = hypothesis["segment_id"] != current_segment_id

                    if is_new_segment:
                        # Finalize old segment before starting new one (only if not already finalized)
                        if current_segment_id is not None and not current_segment_finalized:
                            final_segment = segment_manager_instance.finalize()
                            if final_segment and final_segment["committed"]:
                                print(f"🏁 Finalized: \"{final_segment['committed']}\"")

                                # Layer 3: Append to immutable accumulator
                                finalized = global_accumulator.append(
                                    current_segment_id,
                                    final_segment["committed"]
                                )

                                # Send immutable segment to client
                                await ws.send_json(
                                    {
                                        "type": "segments_finalized",
                                        "segments": [finalized],
                                    }
                                )

                        # Reset for new segment
                        segment_manager_instance = SegmentManager()
                        current_segment_id = hypothesis["segment_id"]
                        current_segment_finalized = False  # Reset flag for new segment
                        print(f"🆕 New segment: {current_segment_id}")

                    # Process hypothesis through segment manager
                    segment = segment_manager_instance.ingest(hypothesis)

                    # Log committed (stable) and partial (unstable) text
                    print(f"✅ Committed: \"{segment['committed']}\"")
                    print(f"⏳ Partial:   \"{segment['partial']}\"")
                    print(f"📊 Revision {segment['revision']}")

                    # Send segment back to client
                    await ws.send_json(
                        {"type": "segments_update", "segments": [segment]}
                    )

                # Reset for next chunk
                pending_metadata = None

    except WebSocketDisconnect:
        print("🔌 Client disconnected: WebSocketDisconnect")
    except RuntimeError as e:
        if "disconnect message has been received" in str(e):
            print("🔌 Client disconnected: Runtime")
        else:
            print(f"❌ WebSocket error: RuntimeError: {e}")
    except Exception as e:
        print(f"❌ WebSocket error: {type(e).__name__}: {e}")
    finally:
        # Cleanup only - no WebSocket sends allowed here
        print("🔌 WebSocket closed")
