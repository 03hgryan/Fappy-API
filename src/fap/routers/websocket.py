# websocket.py
"""
WebSocket handler for streaming ASR.

Architecture:
- ASRAdapter: Normalizes different ASR engines to standard output format
- SegmentManager: Decides word-level stability, emits structured events
- GlobalAccumulator: Stores finalized WordInfo (append-only)

Key invariants:
- WebSocket NEVER diffs strings
- WebSocket NEVER guesses stability
- WebSocket ONLY reacts to finalized_words events
- rendered_text is for UI display only, not authoritative
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fap.asr import create_asr_adapter
from fap.utils.segmentManager import SegmentManager
from fap.utils.globalAccumulator import GlobalAccumulator

router = APIRouter()


@router.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    print("✅ WebSocket connection accepted")

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
    else:
        # Fallback to whisper
        asr_adapter = create_asr_adapter(
            provider="whisper",
            model=shared_model,
        )
    
    print(f"🎙️ Using ASR adapter: {asr_adapter.provider_name}")
    
    # Create SegmentManager with appropriate mode
    asr_mode = "rewriting" if provider == "google" else "incremental"
    segment_manager = SegmentManager(asr_mode=asr_mode)
    global_accumulator = GlobalAccumulator()
    
    pending_metadata = None
    current_segment_id = None

    try:
        while True:
            message = await ws.receive()

            # === Handle metadata (JSON) ===
            if "text" in message:
                import json
                data = json.loads(message["text"])

                # Handle end_stream signal from client
                if data.get("type") == "end_stream":
                    print("🏁 Client requested stream end")
                    
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
                        # Append any remaining finalized words
                        if final_output["finalized_words"]:
                            global_accumulator.append_words(
                                final_output["segment_id"],
                                final_output["finalized_words"]
                            )
                        
                        # Send final segment to client
                        await ws.send_json({
                            "type": "segments_finalized",
                            "segments": [{
                                "segment_id": final_output["segment_id"],
                                "text": final_output["rendered_text"]["stable"],
                                "words": final_output["stable_words"],
                                "final": True,
                            }],
                        })
                        
                        print(f"🏁 Final transcript: \"{final_output['rendered_text']['stable']}\"")
                    
                    # Send full accumulated transcript
                    await ws.send_json({
                        "type": "transcript_final",
                        "transcript": global_accumulator.get_full_transcript(),
                        "words": global_accumulator.get_all_words(),
                        "segments": global_accumulator.get_segments(),
                    })
                    
                    await ws.close()
                    return

                # Regular metadata
                pending_metadata = data
                print(f"📝 Metadata: {pending_metadata}")

            # === Handle audio (binary PCM) ===
            elif "bytes" in message:
                if pending_metadata is None:
                    print("⚠️ Received audio without metadata, skipping")
                    continue

                audio = message["bytes"]
                print(f"🎵 Audio: {len(audio)} bytes, chunk #{pending_metadata.get('chunk_index')}")

                # Feed to ASR adapter
                hypothesis = asr_adapter.feed(audio)

                if hypothesis:
                    # === Detect segment boundary ===
                    is_new_segment = hypothesis["segment_id"] != current_segment_id

                    if is_new_segment and current_segment_id is not None:
                        # Finalize old segment
                        final_output = segment_manager.finalize()
                        
                        if final_output:
                            # Append remaining finalized words
                            if final_output["finalized_words"]:
                                global_accumulator.append_words(
                                    final_output["segment_id"],
                                    final_output["finalized_words"]
                                )
                            
                            # Send finalized segment to client
                            await ws.send_json({
                                "type": "segments_finalized",
                                "segments": [{
                                    "segment_id": final_output["segment_id"],
                                    "text": final_output["rendered_text"]["stable"],
                                    "words": final_output["stable_words"],
                                    "final": True,
                                }],
                            })
                            
                            print(f"🏁 Segment boundary - finalized: \"{final_output['rendered_text']['stable']}\"")
                        
                        # Reset for new segment (preserve asr_mode!)
                        segment_manager = SegmentManager(asr_mode=asr_mode)
                        print(f"🆕 New segment: {hypothesis['segment_id']}")

                    current_segment_id = hypothesis["segment_id"]

                    # === Process through SegmentManager ===
                    segment_output = segment_manager.ingest(hypothesis)

                    # === Handle finalized words (the ONLY authoritative event) ===
                    if segment_output["finalized_words"]:
                        # Append to global accumulator (no diffing!)
                        global_accumulator.append_words(
                            segment_output["segment_id"],
                            segment_output["finalized_words"]
                        )
                        
                        # Log what was finalized
                        finalized_text = " ".join(
                            w["word"] for w in segment_output["finalized_words"]
                        )
                        print(f"🔒 Finalized words: \"{finalized_text}\"")

                    # === Send live update to client ===
                    # This includes both stable and unstable for display
                    await ws.send_json({
                        "type": "segments_update",
                        "segments": [{
                            "segment_id": segment_output["segment_id"],
                            "revision": segment_output["revision"],
                            
                            # Structured word data
                            "stable_words": segment_output["stable_words"],
                            "unstable_words": segment_output["unstable_words"],
                            
                            # Convenience rendering for simple clients
                            "committed": segment_output["rendered_text"]["stable"],
                            "partial": segment_output["rendered_text"]["unstable"],
                            
                            "final": segment_output["final"],
                        }],
                    })

                    # === Logging ===
                    stable = segment_output["rendered_text"]["stable"]
                    unstable = segment_output["rendered_text"]["unstable"]
                    print(f"✅ Stable:   \"{stable}\"")
                    print(f"⏳ Unstable: \"{unstable}\"")
                    print(f"📊 Revision {segment_output['revision']}")

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
        # Cleanup adapter
        if asr_adapter.is_streaming:
            asr_adapter.reset()
        print("🔌 WebSocket closed")