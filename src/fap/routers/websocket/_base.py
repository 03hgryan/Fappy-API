# routers/websocket/_base.py
"""
Shared utilities for WebSocket ASR routes.

Provides common WebSocket message handling, response formatting,
and stream finalization logic.
"""

import json
from typing import Any
from fastapi import WebSocket

from fap.utils.segmentManager import SegmentManager
from fap.utils.globalAccumulator import GlobalAccumulator


async def parse_websocket_message(ws: WebSocket) -> dict[str, Any]:
    """
    Parse incoming WebSocket message.

    Returns:
        dict with either:
        - {"type": "metadata", "data": {...}} for JSON messages
        - {"type": "audio", "data": bytes} for binary audio
        - {"type": "end_stream"} for end stream signal
    """
    message = await ws.receive()

    if "text" in message:
        data = json.loads(message["text"])
        if data.get("type") == "end_stream":
            return {"type": "end_stream"}
        return {"type": "metadata", "data": data}

    elif "bytes" in message:
        return {"type": "audio", "data": message["bytes"]}

    return {"type": "unknown"}


async def send_segment_update(
    ws: WebSocket,
    segment_output: dict,
) -> None:
    """
    Send a segments_update message to the client.

    Args:
        ws: WebSocket connection
        segment_output: Output from SegmentManager.ingest()
    """
    await ws.send_json({
        "type": "segments_update",
        "segments": [{
            "segment_id": segment_output["segment_id"],
            "revision": segment_output["revision"],
            "stable_words": segment_output["stable_words"],
            "unstable_words": segment_output["unstable_words"],
            "committed": segment_output["rendered_text"]["stable"],
            "partial": segment_output["rendered_text"]["unstable"],
            "final": segment_output["final"],
        }],
    })


async def send_segment_finalized(
    ws: WebSocket,
    final_output: dict,
) -> None:
    """
    Send a segments_finalized message to the client.

    Args:
        ws: WebSocket connection
        final_output: Output from SegmentManager.finalize()
    """
    await ws.send_json({
        "type": "segments_finalized",
        "segments": [{
            "segment_id": final_output["segment_id"],
            "text": final_output["rendered_text"]["stable"],
            "words": final_output["stable_words"],
            "final": True,
        }],
    })


async def send_transcript_final(
    ws: WebSocket,
    global_accumulator: GlobalAccumulator,
) -> None:
    """
    Send the final transcript with all accumulated words.

    Args:
        ws: WebSocket connection
        global_accumulator: GlobalAccumulator with all finalized words
    """
    await ws.send_json({
        "type": "transcript_final",
        "transcript": global_accumulator.get_full_transcript(),
        "words": global_accumulator.get_all_words(),
        "segments": global_accumulator.get_segments(),
    })


async def handle_segment_boundary(
    ws: WebSocket,
    segment_manager: SegmentManager,
    global_accumulator: GlobalAccumulator,
    asr_mode: str,
) -> SegmentManager:
    """
    Handle segment boundary - finalize old segment and create new one.

    Args:
        ws: WebSocket connection
        segment_manager: Current SegmentManager
        global_accumulator: GlobalAccumulator for storing finalized words
        asr_mode: ASR mode for new SegmentManager

    Returns:
        New SegmentManager for the next segment
    """
    final_output = segment_manager.finalize()

    if final_output:
        if final_output["finalized_words"]:
            global_accumulator.append_words(
                final_output["segment_id"],
                final_output["finalized_words"]
            )

        await send_segment_finalized(ws, final_output)
        print(f"Segment boundary - finalized: \"{final_output['rendered_text']['stable']}\"")

    return SegmentManager(asr_mode=asr_mode)


async def handle_finalized_words(
    segment_output: dict,
    global_accumulator: GlobalAccumulator,
) -> None:
    """
    Handle finalized words from segment output.

    Args:
        segment_output: Output from SegmentManager.ingest()
        global_accumulator: GlobalAccumulator for storing finalized words
    """
    if segment_output["finalized_words"]:
        global_accumulator.append_words(
            segment_output["segment_id"],
            segment_output["finalized_words"]
        )

        finalized_text = " ".join(
            w["word"] for w in segment_output["finalized_words"]
        )
        print(f"Finalized words: \"{finalized_text}\"")


async def handle_stream_end(
    ws: WebSocket,
    segment_manager: SegmentManager,
    global_accumulator: GlobalAccumulator,
    final_hypothesis: dict | None = None,
) -> None:
    """
    Handle end of stream - finalize everything and send final transcript.

    Args:
        ws: WebSocket connection
        segment_manager: SegmentManager to finalize
        global_accumulator: GlobalAccumulator with all words
        final_hypothesis: Optional final hypothesis from ASR engine
    """
    # Process any final hypothesis
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

    # Send complete transcript
    await send_transcript_final(ws, global_accumulator)
