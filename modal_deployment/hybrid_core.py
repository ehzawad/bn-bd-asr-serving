"""The hybrid FastConformer's inference server, free of Modal and NeMo so it can be tested
offline (test_hybrid_server.py); hybrid_inference_app.py loads the model and wraps it for Modal.

It speaks the slice of vLLM's transcription API the frontend uses, so the frontend treats both
models alike: POST /v1/audio/transcriptions (Bearer key; multipart `file` and `model`) answers
{"text": ...}, and GET /health answers 200. The app is built only after the model has loaded and
warmed up, so /health cannot answer before the model is ready.

Each request carries one piece the frontend has already cut and staged exactly as the benchmark's
asr_core.decode stages it (<= 25 s, 16 kHz mono PCM16 WAV). The piece's bytes are written to a
file unchanged and given to model.transcribe([path], batch_size=1), as asr_core.decode does, and
the text is read the way asr_core.decode reads it. Pieces are decoded one at a time, at batch 1.
"""
import hmac
import io
import json
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

SERVED_MODEL_NAME = "fastconformer-hybrid-bn-bd"
TARGET_SR = 16000
MAX_PIECE_SECONDS = 30  # the frontend sends pieces of at most 25 s (asr_core.CHUNK_SECONDS)
MAX_PIECE_BYTES = 2 * 1024 * 1024  # 30 s of 16 kHz PCM16 is 960 kB
MAX_BODY_BYTES = MAX_PIECE_BYTES + 64 * 1024  # plus multipart framing and the form fields
TRANSCRIBE_PATH = "/v1/audio/transcriptions"


class Guard:
    """Refuse a transcription request without the key, or with an oversized body, before FastAPI
    reads and parses the multipart body: the key is checked from the headers, and the body is
    counted while it arrives."""

    def __init__(self, app, expected_auth: bytes):
        self.app, self.expected = app, expected_auth

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != TRANSCRIBE_PATH:
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])

        async def refuse(status, detail):
            body = json.dumps({"detail": detail}).encode()
            await send({"type": "http.response.start", "status": status,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode())]})
            await send({"type": "http.response.body", "body": body})

        if not hmac.compare_digest(headers.get(b"authorization", b""), self.expected):
            return await refuse(401, "A valid API key is required.")
        length = headers.get(b"content-length")
        if length is not None and length.isdigit() and int(length) > MAX_BODY_BYTES:
            return await refuse(413, "Send pieces of at most 30 s.")
        seen = 0

        async def bounded_receive():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > MAX_BODY_BYTES:
                    raise HTTPException(413, "Send pieces of at most 30 s.")
            return message

        return await self.app(scope, bounded_receive, send)


def wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1.0, 1.0), TARGET_SR, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def create_app(model, api_key: str):
    """model.transcribe(paths, batch_size=1, verbose=False) returns one result per path, as
    NeMo's ASRModel.transcribe does under torch.inference_mode (hybrid_inference_app.py)."""
    expected = f"Bearer {api_key}".encode()
    lock = threading.Lock()  # one decode at a time, at batch 1, as in the benchmark

    def run(wav: bytes) -> str:
        with lock, tempfile.TemporaryDirectory(prefix="stt_bn_") as d:
            path = Path(d) / "chunk_000.wav"
            path.write_bytes(wav)
            results = model.transcribe([str(path)], batch_size=1, verbose=False)
        texts = [(r.text if hasattr(r, "text") else str(r)).strip() for r in results]
        return " ".join(t for t in texts if t)

    run(wav_bytes(np.zeros(TARGET_SR, dtype=np.float32)))  # warm-up, discarded as in the benchmark

    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    api.add_middleware(Guard, expected_auth=expected)

    @api.get("/health")
    async def health():
        return {"status": "ok"}

    @api.post(TRANSCRIBE_PATH)
    async def transcriptions(file: UploadFile = File(...), model_name: str = Form(SERVED_MODEL_NAME, alias="model"),
                             authorization: str = Header("")):
        if not hmac.compare_digest(authorization.encode(), expected):  # Guard checked it already
            raise HTTPException(401, "A valid API key is required.")
        if model_name != SERVED_MODEL_NAME:
            raise HTTPException(404, f"This server serves {SERVED_MODEL_NAME!r} only.")
        data = await file.read(MAX_PIECE_BYTES + 1)
        if len(data) > MAX_PIECE_BYTES:
            raise HTTPException(413, "Send pieces of at most 30 s.")
        try:
            info = sf.info(io.BytesIO(data))
        except Exception:
            raise HTTPException(400, "The piece is not a WAV file.")
        if (info.format, info.subtype, info.samplerate, info.channels) != ("WAV", "PCM_16", TARGET_SR, 1):
            raise HTTPException(400, "Send a 16 kHz mono PCM16 WAV piece.")
        if info.frames > MAX_PIECE_SECONDS * TARGET_SR:
            raise HTTPException(413, "Send pieces of at most 30 s.")
        return {"text": await run_in_threadpool(run, data)}

    return api
