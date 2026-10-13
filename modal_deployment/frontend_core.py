"""The frontend's web app, free of Modal so it can be tested offline (test_frontend.py);
frontend_app.py wraps it for Modal.

Readiness. Nobody can record, upload or transcribe until vLLM's /health has answered
200, which it only does once the engine has fully loaded:
  - /api/status reads Modal's autoscaler stats and never sends a request to the GPU
    container itself, so polling cannot keep the L4 awake. When a container exists but is
    not yet confirmed (it is starting, or this frontend restarted), status starts a
    bounded wake loop that asks /health until it answers;
  - /api/status?probe=1 asks vLLM directly, but never wakes a sleeping service; the page
    uses it right before recording or uploading, and while recording (which keeps the GPU
    warm for that recording);
  - /api/transcribe decodes and checks the upload first (so a bad file never touches the
    GPU), then asks vLLM directly right before sending any audio, and refuses with 503
    (starting the wake) if it does not answer.

Audio follows the benchmark's path unchanged: asr_core.load_audio (16 kHz mono),
asr_core.chunks (<= 25 s pieces cut at pauses, no VAD), each piece staged as a PCM16 WAV
and sent to vLLM's /v1/audio/transcriptions (greedy, language "bn", at most 500 new
tokens), and the pieces' texts joined with single spaces. Uploads are bounded before they
are parsed and before they are decoded, and nothing is stored.
"""
import asyncio
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

SERVED_MODEL_NAME = "voxtral-mini-bn-bd"
MAX_SECONDS = 10 * 60  # longest recording or upload accepted
DURATION_GRACE_S = 5  # codec padding and timer slack on a recording stopped at the limit
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_DECODED_SAMPLES = 64_000_000  # native-rate samples x channels decoded at once (256 MB as float32)
DECODE_CONCURRENCY = 4
INFERENCE_IDLE_MINUTES = 10  # the inference app's scaledown_window
STATS_GRACE_S = 30  # autoscaler stats can lag a container that just answered; trust the answer this long
WAKE_DEADLINE_S = 20 * 60
NOT_READY = "The model is not ready yet. The page unlocks when it is."
# Containers ffmpeg may open for us: audio and video formats only, never playlists or concat
# lists, which could make it read other files.
FFMPEG_FORMATS = {"mov", "mp4", "m4a", "3gp", "3g2", "mj2", "matroska", "webm", "ogg", "mp3", "aac",
                  "wav", "flac", "aiff", "caf", "amr", "wv", "asf", "avi", "flv", "mpegts", "mpeg"}
# asr_core.py is copied verbatim from github.com/ehzawad/bengali-asr-benchmark@5cccd1a
ASR_CORE_SHA256 = "f894ef44ae4834fc80878f8b206e65e8321a0014807559494af0eb8880638a18"


def load_asr_core(path: Path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != ASR_CORE_SHA256:
        raise RuntimeError("asr_core.py is not the benchmark's copy")
    try:
        import torch  # noqa: F401
    except ImportError:  # asr_core needs torch only in decode(), the NeMo path, unused here
        sys.modules["torch"] = types.ModuleType("torch")
    sys.path.insert(0, str(path.parent))
    import asr_core

    return asr_core


class AudioRejected(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


async def gather_or_cancel(*aws):
    """asyncio.gather that cancels the remaining requests when one fails."""
    tasks = [asyncio.ensure_future(a) for a in aws]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class UploadBodyLimit:
    """Refuse an oversized /api/transcribe body while it arrives, before multipart parsing."""

    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/api/transcribe":
            return await self.app(scope, receive, send)
        length = dict(scope["headers"]).get(b"content-length")
        if length is not None and length.isdigit() and int(length) > self.limit:
            body = json.dumps({"detail": "This upload is too large."}).encode()
            await send({"type": "http.response.start", "status": 413,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode())]})
            return await send({"type": "http.response.body", "body": body})
        seen = 0

        async def bounded_receive():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.limit:
                    raise HTTPException(413, "This upload is too large.")
            return message

        return await self.app(scope, bounded_receive, send)


def create_app(serve_fn, api_key: str, asr_core, static_dir: Path, *, transport=None,
               wake_probe_timeout=170.0, wake_retry_s=2.0, wake_deadline_s=WAKE_DEADLINE_S,
               stats_grace_s=STATS_GRACE_S, max_upload_bytes=MAX_UPLOAD_BYTES,
               max_decoded_samples=MAX_DECODED_SAMPLES):
    """serve_fn is the inference app's web function: get_web_url() and get_current_stats()
    are all this uses (as their .aio forms). transport lets tests stand in for vLLM."""
    auth = {"Authorization": f"Bearer {api_key}"}
    # Modal answers a request still waiting after 150 s (e.g. on a cold start) with a 303 to
    # a result URL that blocks until the response is ready, so follow redirects.
    client = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=30), follow_redirects=True,
                               limits=httpx.Limits(max_connections=64), transport=transport)
    # ready: vLLM's /health answered 200, as of ready_at (monotonic seconds)
    st = {"url": None, "ready": False, "ready_at": 0.0, "wake": None, "wake_started": None,
          "wake_failed": False}
    decode_slots = asyncio.Semaphore(DECODE_CONCURRENCY)
    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    api.add_middleware(UploadBodyLimit, limit=max_upload_bytes + 1024 * 1024)  # + multipart framing

    async def base_url():
        if st["url"] is None:
            st["url"] = await serve_fn.get_web_url.aio()
        return st["url"]

    def mark_ready():
        st["ready"], st["ready_at"], st["wake_failed"] = True, time.monotonic(), False

    async def probe(timeout=3.0):
        """Ask vLLM directly. A container that is still starting does not answer in time."""
        try:
            r = await client.get(f"{await base_url()}/health", timeout=timeout)
            ok = r.status_code == 200
        except httpx.HTTPError:
            ok = False
        if ok:
            mark_ready()
        else:
            st["ready"] = False
        return ok

    async def wake_loop():
        # Each /health request waits on the cold start; retry until the engine answers.
        deadline = time.monotonic() + wake_deadline_s
        while time.monotonic() < deadline:
            if await probe(timeout=wake_probe_timeout):
                return
            await asyncio.sleep(wake_retry_s)
        st["wake_failed"] = True  # stop retrying on our own, so a broken container can scale down

    def waking():
        return st["wake"] is not None and not st["wake"].done()

    def start_wake():
        if not waking():
            st["wake_started"] = time.monotonic()
            st["wake"] = asyncio.create_task(wake_loop())

    async def refresh(force_probe: bool):
        stats = await serve_fn.get_current_stats.aio()
        just_confirmed = st["ready"] and time.monotonic() - st["ready_at"] < stats_grace_s
        if stats.num_total_runners == 0 and not just_confirmed:
            st["ready"] = False  # scaled to zero since the last check
        elif force_probe:  # never sent to a sleeping service, so it cannot wake one
            await probe()
        return stats

    async def ensure_ready():
        """Ask vLLM now; if it does not answer, start waking it and refuse."""
        if not await probe():
            start_wake()
            raise HTTPException(503, NOT_READY)

    async def state_now(check: bool):
        stats = await refresh(force_probe=check)
        exists = stats.num_total_runners > 0 or stats.backlog > 0
        if exists and not st["ready"] and not waking() and not st["wake_failed"]:
            start_wake()  # starting, or this frontend restarted: confirm it with a bounded loop
        if st["ready"]:
            state = "ready"
        elif waking() or (exists and not st["wake_failed"]):
            state = "waking"
        elif st["wake_failed"]:
            state = "error"
        else:
            state = "asleep"
        waited = (round(time.monotonic() - st["wake_started"])
                  if state == "waking" and st["wake_started"] else None)
        return {"state": state, "waking_for_s": waited, "max_seconds": MAX_SECONDS,
                "idle_minutes": INFERENCE_IDLE_MINUTES, "model": "Voxtral Mini 3B bn-BD", "gpu": "NVIDIA L4"}

    @api.get("/api/status")
    async def status(check: bool = Query(False, alias="probe")):  # ?probe=1 asks vLLM directly
        return await state_now(check)

    @api.post("/api/wake")
    async def wake():
        await refresh(force_probe=False)  # clears a stale "ready" if the GPU scaled to zero
        if not st["ready"]:
            st["wake_failed"] = False  # an explicit wake retries even after a failed one
            start_wake()
        return await state_now(False)

    def checked_load(source):
        """asr_core.load_audio, after bounding duration and decoded size from the header."""
        info = sf.info(source)
        if info.frames > (MAX_SECONDS + DURATION_GRACE_S) * info.samplerate:
            raise AudioRejected(413, f"Audio up to {MAX_SECONDS // 60} minutes is accepted "
                                     f"(this is {info.frames / info.samplerate / 60:.1f}).")
        if info.frames * info.channels > max_decoded_samples:
            raise AudioRejected(413, "This audio is too large to process. "
                                     "Please use a shorter file or a lower sample rate.")
        if hasattr(source, "seek"):
            source.seek(0)
        return asr_core.load_audio(source)

    def decode(data: bytes):
        """16 kHz mono float32 via asr_core.load_audio. Containers libsndfile cannot read
        (WebM, MP4/M4A, ...) are first decoded by ffmpeg to a float WAV at the source rate and
        channel count, so downmixing and resampling still happen in asr_core."""
        buf = io.BytesIO(data)
        try:
            sf.info(buf)
            native = True
        except Exception:
            native = False
        if native:
            buf.seek(0)
            return checked_load(buf)
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "upload", Path(d) / "decoded.wav"
            src.write_bytes(data)
            p = subprocess.run(["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-show_entries",
                                "format=format_name", "-of", "default=nw=1:nk=1", str(src)],
                               capture_output=True, text=True, timeout=30)
            if p.returncode != 0:
                raise AudioRejected(415, "This file could not be read as audio.")
            if not set(p.stdout.strip().split(",")) & FFMPEG_FORMATS:
                raise AudioRejected(415, "This file type is not supported.")
            max_bytes = max_decoded_samples * 4
            r = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-i", str(src),
                 "-map", "0:a:0", "-t", str(MAX_SECONDS + DURATION_GRACE_S + 1), "-fs", str(max_bytes),
                 "-c:a", "pcm_f32le", "-f", "wav", str(dst)],
                capture_output=True, timeout=180)
            if r.returncode != 0 or not dst.exists():
                raise AudioRejected(415, "This file could not be read as audio.")
            if dst.stat().st_size >= max_bytes - 1024 * 1024:  # stopped by -fs
                raise AudioRejected(413, "This audio is too large to process. "
                                         "Please use a shorter file or a lower sample rate.")
            return checked_load(str(dst))

    def staged(piece) -> bytes:
        # asr_core.decode's staging: clip, then PCM16 16 kHz WAV
        buf = io.BytesIO()
        sf.write(buf, np.clip(piece, -1.0, 1.0), asr_core.TARGET_SR, subtype="PCM_16", format="WAV")
        return buf.getvalue()

    async def transcribe_piece(wav: bytes) -> str:
        r = await client.post(
            f"{await base_url()}/v1/audio/transcriptions", headers=auth,
            files={"file": ("piece.wav", wav, "audio/wav")},
            data={"model": SERVED_MODEL_NAME, "language": "bn", "temperature": "0",
                  "max_completion_tokens": "500", "response_format": "json"})
        r.raise_for_status()
        return r.json()["text"].strip()

    @api.post("/api/transcribe")
    async def transcribe(file: UploadFile = File(...)):
        if file.size is not None and file.size > max_upload_bytes:
            raise HTTPException(413, "This upload is too large.")
        data = await file.read()
        if not data:
            raise HTTPException(400, "The upload was empty.")
        if len(data) > max_upload_bytes:
            raise HTTPException(413, "This upload is too large.")
        t0 = time.perf_counter()
        try:
            async with decode_slots:
                audio, note = await run_in_threadpool(decode, data)
        except AudioRejected as e:
            raise HTTPException(e.status, e.detail)
        except subprocess.TimeoutExpired:
            raise HTTPException(422, "Decoding this file took too long.")
        del data
        seconds = len(audio) / asr_core.TARGET_SR
        if seconds < 0.1:
            raise HTTPException(400, "No audio found in this file.")
        pieces, forced = asr_core.chunks(audio)
        wavs = [staged(p) for p in pieces]
        del audio, pieces
        await ensure_ready()  # only valid audio may wake the GPU, and only right before it is sent
        t1 = time.perf_counter()
        try:
            texts = await gather_or_cancel(*(transcribe_piece(w) for w in wavs))
        except httpx.HTTPError as e:
            raise HTTPException(502, f"The model service did not answer ({type(e).__name__}). "
                                     "Please try again in a moment.")
        mark_ready()
        return {"text": " ".join(t for t in texts if t), "pieces": texts,
                "audio_seconds": round(seconds, 2), "forced_cuts": forced, "note": note,
                "prep_seconds": round(t1 - t0, 2),
                "model_seconds": round(time.perf_counter() - t1, 2)}

    @api.get("/")
    async def index():
        return FileResponse(Path(static_dir) / "index.html", headers={"Cache-Control": "no-cache"})

    return api
