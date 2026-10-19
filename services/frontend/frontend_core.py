"""The frontend's web app, free of Modal so it can be tested offline (test_frontend.py);
modal_app.py wraps it for Modal.

Models. The page offers each model in MODELS; each is its own inference service (a Modal app
under services/) on its own GPU, woken and tracked separately, and every endpoint takes ?model=
(a form field for /api/transcribe). Only the model the page has selected is woken. Every
inference service answers the same contract, so this app treats them alike:
  - GET /health: 200 only once the model is ready;
  - POST /v1/audio/transcriptions (Authorization: Bearer VLLM_API_KEY): one <= 25 s piece as a
    16 kHz PCM16 WAV, with the OpenAI form fields sent by transcribe_piece, answering {"text"};
  - its Modal web endpoint is named "serve" (a web function, or the `serve` method of the class named by
    the model's "cls", for a snapshotted service).

Readiness. Nobody can record, upload or transcribe until the model's /health has answered
200, which it only does once the model has fully loaded. For a model with warm_up set
(Voxtral), an explicit wake also waits for one short warm-up transcription, because a fresh
vLLM spends about a second more on its first request.
  - /api/status reads Modal's autoscaler stats and never wakes a sleeping model, so polling
    cannot keep the GPU awake. Only when a container already exists but is not yet confirmed
    (it is starting, or this frontend restarted) does status start a wake loop that asks
    /health until it answers; that loop never sends a warm-up;
  - a wake loop runs at most WAKE_DEADLINE_S, and stops early once nobody has asked about its
    model for INTEREST_S (the page polls the selected model every 2 s while it wakes), so a
    model the page switched away from, or a closed tab, does not keep a GPU busy;
  - /api/status?probe=1 asks the model directly, but never wakes a sleeping service; the
    page uses it right before recording or uploading, and while recording (which keeps the
    GPU warm for that recording);
  - /api/transcribe decodes and checks the upload first (so a bad file never touches the
    GPU), then asks the model directly right before sending any audio, and refuses with 503
    (starting the wake) if it does not answer.

Audio follows the benchmark's path unchanged for every model: asr_core.load_audio (16 kHz
mono), asr_core.chunks (<= 25 s pieces cut at pauses, no VAD), each piece staged as a PCM16
WAV and sent to the model's /v1/audio/transcriptions, and the pieces' texts joined with single
spaces. Voxtral Mini (FP8 decoder) decodes in vLLM (greedy, language "bn", at most 500 new tokens). Uploads are
bounded before they are parsed and before they are decoded, and nothing is stored.
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
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, RedirectResponse

# One inference service per model; `app` is its Modal app, whose web endpoint is named "serve" (a method of
# class `cls` when the service is a snapshotted class).
# To offer another model, deploy its service and add it here and to MODELS in static/index.html.
# warm_up: after an explicit wake the frontend sends one warm-up piece (vLLM's first request is
# slow); a service that warms itself up before it answers /health sets False.
MODELS = {
    "voxtral": {"label": "Voxtral Mini 3B bn-BD (FP8)", "short": "Voxtral Mini", "app": "voxtral-mini-bn-inference",
                "cls": "Voxtral",  # snapshotted class; its web endpoint is the `serve` method
                "served_model": "voxtral-mini-bn-bd", "engine": "vLLM", "gpu": "NVIDIA L4",
                "cold_start": "about half a minute", "warm_up": True},
}
DEFAULT_MODEL = "voxtral"
MAX_SECONDS = 10 * 60  # longest recording or upload accepted
DURATION_GRACE_S = 5  # codec padding and timer slack on a recording stopped at the limit
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_DECODED_SAMPLES = 64_000_000  # native-rate samples x channels decoded at once (256 MB as float32)
DECODE_CONCURRENCY = 4
INFERENCE_IDLE_MINUTES = 10  # the inference app's scaledown_window
STATS_GRACE_S = 30  # autoscaler stats can lag a container that just answered; trust the answer this long
WAKE_DEADLINE_S = 20 * 60
INTEREST_S = 60  # a wake loop stops once nobody has asked about its model for this long
NOT_READY = "The model is not ready yet. The page unlocks when it is."
# Containers ffmpeg may open for us: audio and video formats only, never playlists or concat
# lists, which could make it read other files. -format_whitelist refuses anything else before the
# input is opened.
FFMPEG_FORMATS = {"mov", "mp4", "m4a", "3gp", "3g2", "mj2", "matroska", "webm", "ogg", "mp3", "aac",
                  "wav", "flac", "aiff", "caf", "amr", "wv", "asf", "avi", "flv", "mpegts", "mpeg"}
FFMPEG_INPUT = ["-protocol_whitelist", "file", "-format_whitelist", ",".join(sorted(FFMPEG_FORMATS))]
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


def warm_up_wav(seconds=2.0) -> bytes:
    """A fixed, quiet noise piece (16 kHz PCM16 WAV) for the warm-up transcription."""
    noise = np.random.default_rng(0).normal(0.0, 0.01, int(seconds * 16000)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, noise, 16000, subtype="PCM_16", format="WAV")
    return buf.getvalue()


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


class Backend:
    """One model's inference app, and its readiness as this frontend last saw it. serve_fn is the
    app's web function: get_web_url() and get_current_stats() are all this uses (as their .aio
    forms)."""

    def __init__(self, serve_fn, client, *, wake_probe_timeout, wake_retry_s, wake_deadline_s, stats_grace_s,
                 interest_s, warm_up=None):
        self.serve_fn, self.client, self.warm_up = serve_fn, client, warm_up
        self.interest_s, self.last_interest = interest_s, time.monotonic()
        self.wake_probe_timeout, self.wake_retry_s = wake_probe_timeout, wake_retry_s
        self.wake_deadline_s, self.stats_grace_s = wake_deadline_s, stats_grace_s
        self.url = None
        self.ready, self.ready_at = False, 0.0  # /health answered 200, as of ready_at (monotonic seconds)
        self.wake, self.wake_started, self.wake_failed = None, None, False
        self.want_warm = False  # an explicit wake asked for a warm-up once /health answers

    async def base_url(self):
        if self.url is None:
            self.url = await self.serve_fn.get_web_url.aio()
        return self.url

    def mark_ready(self):
        self.ready, self.ready_at, self.wake_failed = True, time.monotonic(), False

    async def healthy(self, timeout):
        try:
            r = await self.client.get(f"{await self.base_url()}/health", timeout=timeout)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def probe(self, timeout=3.0):
        """Ask the model directly. A container that is still starting does not answer in time."""
        ok = await self.healthy(timeout)
        if ok:
            self.mark_ready()
        else:
            self.ready = False
        return ok

    def touch(self):
        """Someone (the page polling, a wake, an upload) asked about this model just now."""
        self.last_interest = time.monotonic()

    async def wake_loop(self):
        # Each /health request waits on the cold start; retry until the model answers, then warm it up.
        deadline = time.monotonic() + self.wake_deadline_s
        while time.monotonic() < deadline:
            if await self.healthy(timeout=self.wake_probe_timeout):
                warm, self.want_warm = self.want_warm, False
                if warm and self.warm_up is not None:
                    try:  # its text is discarded; a failure only loses the speed-up
                        await self.warm_up(self)
                    except Exception:
                        pass
                self.mark_ready()
                return
            self.ready = False
            if time.monotonic() - self.last_interest > self.interest_s:
                return  # nobody is waiting for this model any more; the next status or wake starts over
            await asyncio.sleep(self.wake_retry_s)
        self.wake_failed = True  # stop retrying on our own, so a broken container can scale down

    def waking(self):
        return self.wake is not None and not self.wake.done()

    def start_wake(self, warm=False):
        self.want_warm = self.want_warm or warm  # also reaches a loop that is already running
        if not self.waking():
            self.wake_started = time.monotonic()
            self.wake = asyncio.create_task(self.wake_loop())

    async def refresh(self, force_probe: bool):
        stats = await self.serve_fn.get_current_stats.aio()
        just_confirmed = self.ready and time.monotonic() - self.ready_at < self.stats_grace_s
        if stats.num_total_runners == 0 and not just_confirmed:
            self.ready = False  # scaled to zero since the last check
        elif force_probe:  # never sent to a sleeping service, so it cannot wake one
            await self.probe()
        return stats

    async def ensure_ready(self):
        """Ask the model now; if it does not answer, start waking it and refuse."""
        if not await self.probe():
            self.start_wake()
            raise HTTPException(503, NOT_READY)

    async def wake_now(self):
        await self.refresh(force_probe=False)  # clears a stale "ready" if the GPU scaled to zero
        if not self.ready:
            self.wake_failed = False  # an explicit wake retries even after a failed one
            self.start_wake(warm=True)

    async def state_now(self, check: bool):
        stats = await self.refresh(force_probe=check)
        exists = stats.num_total_runners > 0 or stats.backlog > 0
        if exists and not self.ready and not self.waking() and not self.wake_failed:
            self.start_wake()  # starting, or this frontend restarted: confirm it with a bounded loop
        if self.ready:
            state = "ready"
        elif self.waking() or (exists and not self.wake_failed):
            state = "waking"
        elif self.wake_failed:
            state = "error"
        else:
            state = "asleep"
        waited = (round(time.monotonic() - self.wake_started)
                  if state == "waking" and self.wake_started else None)
        return state, waited


def create_app(serve_fns: dict, api_key: str, asr_core, static_dir: Path, *, transport=None,
               wake_probe_timeout=170.0, wake_retry_s=2.0, wake_deadline_s=WAKE_DEADLINE_S,
               stats_grace_s=STATS_GRACE_S, interest_s=INTEREST_S, max_upload_bytes=MAX_UPLOAD_BYTES,
               max_decoded_samples=MAX_DECODED_SAMPLES, voice_chat_url=None):
    """serve_fns maps each MODELS key to its inference app's web function (see Backend).
    transport lets tests stand in for the inference apps."""
    auth = {"Authorization": f"Bearer {api_key}"}
    # Modal answers a request still waiting after 150 s (e.g. on a cold start) with a 303 to
    # a result URL that blocks until the response is ready, so follow redirects.
    client = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=30), follow_redirects=True,
                               limits=httpx.Limits(max_connections=64), transport=transport)
    async def transcribe_piece(b: "Backend", served_model: str, wav: bytes, name="piece.wav") -> str:
        r = await client.post(
            f"{await b.base_url()}/v1/audio/transcriptions", headers=auth,
            files={"file": (name, wav, "audio/wav")},
            data={"model": served_model, "language": "bn", "temperature": "0",
                  "max_completion_tokens": "500", "response_format": "json"})
        r.raise_for_status()
        return r.json()["text"].strip()

    warm_wav = warm_up_wav()

    def warm_up_for(served_model):
        async def warm_up(b):
            await transcribe_piece(b, served_model, warm_wav, name="warm-up.wav")
        return warm_up

    backends = {name: Backend(fn, client, wake_probe_timeout=wake_probe_timeout, wake_retry_s=wake_retry_s,
                              wake_deadline_s=wake_deadline_s, stats_grace_s=stats_grace_s,
                              interest_s=interest_s,
                              warm_up=warm_up_for(MODELS[name]["served_model"]) if MODELS[name]["warm_up"] else None)
                for name, fn in serve_fns.items() if name in MODELS}
    decode_slots = asyncio.Semaphore(DECODE_CONCURRENCY)
    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    api.add_middleware(UploadBodyLimit, limit=max_upload_bytes + 1024 * 1024)  # + multipart framing

    def backend(name: str) -> Backend:
        if name not in backends:
            raise HTTPException(400, "Unknown model.")
        backends[name].touch()
        return backends[name]

    async def status_body(name: str, check: bool):
        state, waited = await backend(name).state_now(check)
        m = MODELS[name]
        return {"state": state, "waking_for_s": waited, "max_seconds": MAX_SECONDS,
                "idle_minutes": INFERENCE_IDLE_MINUTES, "model": name, "label": m["label"],
                "engine": m["engine"], "gpu": m["gpu"], "cold_start": m["cold_start"]}

    @api.get("/api/status")
    async def status(model: str = Query(DEFAULT_MODEL), check: bool = Query(False, alias="probe")):
        return await status_body(model, check)  # ?probe=1 asks the model directly

    @api.post("/api/wake")
    async def wake(model: str = Query(DEFAULT_MODEL)):
        await backend(model).wake_now()
        return await status_body(model, False)

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
            p = subprocess.run(["ffprobe", "-v", "error", *FFMPEG_INPUT, "-show_entries",
                                "format=format_name", "-of", "default=nw=1:nk=1", str(src)],
                               capture_output=True, text=True, timeout=30)
            if p.returncode != 0:
                raise AudioRejected(415, "This file could not be read as audio.")
            if not set(p.stdout.strip().split(",")) & FFMPEG_FORMATS:
                raise AudioRejected(415, "This file type is not supported.")
            max_bytes = max_decoded_samples * 4
            r = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", *FFMPEG_INPUT, "-i", str(src),
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

    @api.post("/api/transcribe")
    async def transcribe(file: UploadFile = File(...), model: str = Form(DEFAULT_MODEL)):
        b = backend(model)
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
        await b.ensure_ready()  # only valid audio may wake the GPU, and only right before it is sent
        t1 = time.perf_counter()
        served = MODELS[model]["served_model"]
        try:
            texts = await gather_or_cancel(*(transcribe_piece(b, served, w) for w in wavs))
        except httpx.HTTPError as e:
            raise HTTPException(502, f"The model service did not answer ({type(e).__name__}). "
                                     "Please try again in a moment.")
        b.mark_ready()
        return {"text": " ".join(t for t in texts if t), "pieces": texts, "model": model,
                "audio_seconds": round(seconds, 2), "forced_cuts": forced, "note": note,
                "prep_seconds": round(t1 - t0, 2),
                "model_seconds": round(time.perf_counter() - t1, 2)}

    @api.get("/")
    async def index():
        return FileResponse(Path(static_dir) / "index.html", headers={"Cache-Control": "no-cache"})

    voice_chat = {}

    @api.get("/chat")
    async def chat():
        """The Bangla voice chatbot (a separate app, bn-voice-chat): /chat on this page forwards there.
        voice_chat_url is a URL or a callable returning one, resolved once."""
        if voice_chat_url is None:
            raise HTTPException(404, "Not found.")
        if "url" not in voice_chat:
            try:
                url = voice_chat_url() if callable(voice_chat_url) else voice_chat_url
                voice_chat["url"] = await url if asyncio.iscoroutine(url) else url
            except Exception:
                raise HTTPException(503, "The voice chat is not reachable right now.")
        return RedirectResponse(voice_chat["url"], status_code=307)

    return api
