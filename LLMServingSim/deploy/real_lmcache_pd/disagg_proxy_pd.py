"""Minimal HTTP proxy for the LMCache v1 push-style P/D flow."""

from __future__ import annotations

import argparse
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--prefiller-host", required=True)
    parser.add_argument("--prefiller-port", type=int, default=8100)
    parser.add_argument("--decoder-host", required=True)
    parser.add_argument("--decoder-port", type=int, default=8200)
    parser.add_argument("--receiver-init-port", type=int, default=55555)
    parser.add_argument("--receiver-alloc-port", type=int, default=55556)
    parser.add_argument("--receiver-query-port", type=int, default=55557)
    return parser.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prefill = httpx.AsyncClient(
        base_url=f"http://{app.state.args.prefiller_host}:{app.state.args.prefiller_port}/v1",
        timeout=None,
    )
    app.state.decode = httpx.AsyncClient(
        base_url=f"http://{app.state.args.decoder_host}:{app.state.args.decoder_port}/v1",
        timeout=None,
    )
    yield
    await app.state.prefill.aclose()
    await app.state.decode.aclose()


app = FastAPI(title="LMCache P/D proxy", lifespan=lifespan)


async def handle(request: Request, endpoint: str):
    args = request.app.state.args
    payload = await request.json()
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))

    prefill_payload = dict(payload)
    prefill_payload["stream"] = False
    prefill_payload["max_tokens"] = 1
    prefill_payload.pop("max_completion_tokens", None)
    prefill_payload["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "disagg_spec": {
            "req_id": request_id,
            "receiver_host": args.decoder_host,
            "receiver_init_port": [args.receiver_init_port],
            "receiver_alloc_port": [args.receiver_alloc_port],
            "receiver_query_port": [args.receiver_query_port],
        },
    }
    headers = {"X-Request-Id": request_id}
    prefill_response = await request.app.state.prefill.post(
        endpoint,
        json=prefill_payload,
        headers=headers,
    )
    prefill_response.raise_for_status()

    decode_payload = dict(payload)
    decode_payload.pop("kv_transfer_params", None)
    if payload.get("stream", False):
        async def stream():
            async with request.app.state.decode.stream(
                "POST", endpoint, json=decode_payload, headers=headers
            ) as streamed:
                streamed.raise_for_status()
                async for chunk in streamed.aiter_bytes():
                    yield chunk

        return StreamingResponse(stream(), media_type="text/event-stream")
    response = await request.app.state.decode.post(
        endpoint,
        json=decode_payload,
        headers=headers,
    )
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type"),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/completions")
async def completions(request: Request):
    return await handle(request, "/completions")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await handle(request, "/chat/completions")


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    app.state.args = args
    uvicorn.run(app, host=args.host, port=args.port)
