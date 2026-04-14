"""WebSocket endpoint for streaming training metrics."""
import asyncio
import json
from fastapi import WebSocket, WebSocketDisconnect
from modules.run_manager import get_active_run


async def training_websocket(websocket: WebSocket, run_id: str):
    await websocket.accept()
    try:
        run = get_active_run()
        if run is None or run.run_id != run_id:
            await websocket.send_json({"type": "error", "message": "Run not found"})
            await websocket.close()
            return

        # Stream metrics from the queue
        while True:
            # Check if there's a message (non-blocking)
            try:
                msg = run.metrics_queue.get_nowait()
                await websocket.send_json(msg)
                if msg.get("type") in ("completed", "cancelled", "error"):
                    break
            except Exception:
                # No message yet, check if run is still alive
                if run.status in ("completed", "cancelled", "error") and run.metrics_queue.empty():
                    break
                # Yield control
                await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
