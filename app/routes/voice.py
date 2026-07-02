from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from app.core.call_handler import handle_call

router = APIRouter(prefix="/voice", tags=["voice"])


@router.post("/incoming")
async def incoming_call(request: Request) -> Response:
    host = request.headers.get("host") or request.url.hostname
    stream_url = f"wss://{host}/voice/stream"
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Connect><Stream url="{stream_url}" /></Connect>'
        "</Response>"
    )
    return Response(content=twiml, media_type="application/xml")


@router.websocket("/stream")
async def media_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        await handle_call(websocket)
    except WebSocketDisconnect:
        pass
