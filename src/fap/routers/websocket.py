#websocket.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fap.asr import StreamingASR
from fap.utils.segmentManager import SegmentManager
from fap.utils.globalAccumulator import GlobalAccumulator

router = APIRouter()


def extract_overlap_delta(prev: str, curr: str) -> str | None:
    """
    Return the non-overlapping suffix of curr compared to prev.

    Finds the longest suffix of prev that matches a prefix of curr,
    then returns the new suffix. This handles rolling buffer shifts.

    Args:
        prev: Previous committed text
        curr: Current committed text

    Returns:
        The new delta suffix, or None if no safe overlap found

    Example:
        prev = "Hey, Vsauce"
        curr = "Vsauce, Michael here"
        Returns: ", Michael here" (overlap on "Vsauce")
    """
    if not prev:
        return curr.strip() if curr.strip() else None

    max_overlap = min(len(prev), len(curr))

    # Find longest suffix of prev that is prefix of curr
    for i in range(max_overlap, 0, -1):
        if prev[-i:] == curr[:i]:
            delta = curr[i:].strip()
            return delta if delta else None

    # No overlap → unsafe to append (rolling buffer dropped everything)
    return None


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
    last_committed_text = ""  # Track last committed text for overlap-based delta extraction

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

                    # Finalize active segment - only append remaining delta
                    final_segment = segment_manager_instance.finalize()
                    if final_segment and final_segment["committed"]:
                        print(f"🏁 End stream - final committed: \"{final_segment['committed']}\"")

                        # Only append remaining delta (what hasn't been incrementally accumulated)
                        final_text = final_segment["committed"]
                        delta = extract_overlap_delta(last_committed_text, final_text)

                        if delta:
                            finalized = global_accumulator.append(
                                current_segment_id,
                                delta
                            )

                            await ws.send_json({
                                "type": "segments_finalized",
                                "segments": [finalized],
                            })

                            print(f"🔒 Final delta at end: \"{delta}\"")
                        else:
                            print("✅ Nothing new to finalize (already accumulated incrementally)")

                    # 🔥 SEND FULL ACCUMULATED TRANSCRIPT
                    await ws.send_json({
                        "type": "transcript_final",
                        "transcript": global_accumulator.get_full_transcript(),
                        "segments": global_accumulator.get_segments(),
                    })

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
                        # Finalize old segment before starting new one
                        if current_segment_id is not None:
                            final_segment = segment_manager_instance.finalize()
                            if final_segment and final_segment["committed"]:
                                print(f"🏁 Segment boundary - final committed: \"{final_segment['committed']}\"")

                                # Only append remaining delta (what hasn't been incrementally accumulated)
                                final_text = final_segment["committed"]
                                delta = extract_overlap_delta(last_committed_text, final_text)

                                if delta:
                                    finalized = global_accumulator.append(
                                        current_segment_id,
                                        delta
                                    )

                                    await ws.send_json({
                                        "type": "segments_finalized",
                                        "segments": [finalized],
                                    })

                                    print(f"🔒 Final delta at boundary: \"{delta}\"")
                                else:
                                    print("✅ Nothing new to finalize (already accumulated incrementally)")

                        # Reset for new segment
                        segment_manager_instance = SegmentManager()
                        current_segment_id = hypothesis["segment_id"]
                        last_committed_text = ""  # Reset for new segment
                        print(f"🆕 New segment: {current_segment_id}")

                    # Process hypothesis through segment manager
                    segment = segment_manager_instance.ingest(hypothesis)

                    # 🔥 INCREMENTAL DELTA ACCUMULATION (Overlap-based)
                    # Find longest common suffix/prefix overlap, append only new suffix
                    # This handles rolling buffer shifts/rephrases correctly
                    new_committed = segment["committed"]

                    delta = extract_overlap_delta(last_committed_text, new_committed)

                    if delta:
                        # Append delta to immutable accumulator
                        finalized = global_accumulator.append(
                            current_segment_id,
                            delta
                        )

                        # Send finalized delta to client
                        await ws.send_json({
                            "type": "segments_finalized",
                            "segments": [finalized],
                        })

                        print(f"🔒 Finalized delta: \"{delta}\"")

                        # Update tracking
                        last_committed_text = new_committed
                    else:
                        # No safe overlap detected (rolling buffer dropped everything or no change)
                        print("⏸️  No safe overlap delta detected")

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