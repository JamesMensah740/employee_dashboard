# server.py
import os
import asyncio
from typing import Dict

import httpx
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

UPSTREAM_HOST = os.getenv("STREAMLIT_HOST", "127.0.0.1")
UPSTREAM_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
UPSTREAM_HTTP = f"http://{UPSTREAM_HOST}:{UPSTREAM_PORT}"
UPSTREAM_WS = f"ws://{UPSTREAM_HOST}:{UPSTREAM_PORT}"

app = FastAPI(title="Streamlit Proxy")

@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok", status_code=200)

def _filter_headers(h: Dict[str, str]) -> Dict[str, str]:
    drop = {"host","connection","keep-alive","proxy-authenticate","proxy-authorization","te",
            "trailers","transfer-encoding","upgrade"}
    return {k: v for k, v in h.items() if k.lower() not in drop}

@app.api_route("/{path:path}", methods=["GET","POST","PUT","PATCH","DELETE","HEAD","OPTIONS"])
async def proxy_http(request: Request, path: str):
    target = f"{UPSTREAM_HTTP}/" + path
    if request.url.query:
        target += f"?{request.url.query}"
    headers = _filter_headers(dict(request.headers))
    body = await request.body()
    async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
        upstream_resp = await client.request(request.method, target, headers=headers, content=body)
    resp_headers = _filter_headers(dict(upstream_resp.headers))
    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers=resp_headers,
        media_type=upstream_resp.headers.get("content-type"),
    )

@app.websocket("/{path:path}")
async def proxy_ws(websocket: WebSocket, path: str):
    await websocket.accept()
    qs = websocket.url.query
    upstream_url = f"{UPSTREAM_WS}/" + path + (f"?{qs}" if qs else "")
    import websockets
    try:
        async with websockets.connect(upstream_url, max_size=None) as upstream:
            async def client_to_upstream():
                try:
                    while True:
                        msg = await websocket.receive()
                        if "text" in msg:
                            await upstream.send(msg["text"])
                        elif "bytes" in msg:
                            await upstream.send(msg["bytes"])
                except WebSocketDisconnect:
                    await upstream.close()
                except Exception:
                    await upstream.close()

            async def upstream_to_client():
                try:
                    while True:
                        msg = await upstream.recv()
                        if isinstance(msg, (bytes, bytearray)):
                            await websocket.send_bytes(msg)
                        else:
                            await websocket.send_text(msg)
                except Exception:
                    try:
                        await websocket.close()
                    except Exception:
                        pass

            await asyncio.gather(client_to_upstream(), upstream_to_client())
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
