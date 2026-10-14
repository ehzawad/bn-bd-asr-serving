#!/usr/bin/env python
"""Offline check of the frontend's readiness gating, model selection and upload handling, against
fake Modal autoscalers and fake inference servers (no Modal, no GPU). Model selection is checked
with a test-only second model whose server, like any service with warm_up False, needs no warm-up.
Needs: fastapi, httpx, python-multipart, numpy, soundfile, librosa, and ffmpeg/ffprobe on PATH.

    python services/frontend/test_frontend.py
"""
import asyncio
import io
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
SHARED = HERE.parents[1] / "shared"
sys.path.insert(0, str(HERE))
import frontend_core  # noqa: E402

# A second inference service, registered for these tests only.
frontend_core.MODELS["other"] = {**frontend_core.MODELS["voxtral"], "label": "Other test model", "short": "Other",
                                 "app": "other-test-inference", "served_model": "other-test-model", "warm_up": False}


class Fake:
    """A Modal web function (autoscaler stats) plus the inference server behind it."""

    def __init__(self, host="inference.test"):
        self.host = host
        self.models_seen = []
        self.runners = 0  # containers, starting or up
        self.up = False  # the engine has loaded and /health answers 200
        self.health_calls = 0
        self.transcriptions = 0
        self.warmups = 0  # warm-up transcriptions sent by the wake loop, counted apart
        self.fail_next_piece = False
        self.piece_delay = 0.0
        self.inflight = 0
        self.get_web_url = types.SimpleNamespace(aio=self._url)
        self.get_current_stats = types.SimpleNamespace(aio=self._stats)

    async def _url(self):
        return f"http://{self.host}"

    async def _stats(self):
        return types.SimpleNamespace(num_total_runners=self.runners, backlog=0)

    async def handler(self, request: httpx.Request):
        self.runners = max(self.runners, 1)  # any request to a sleeping app starts a container
        if request.url.path == "/health":
            self.health_calls += 1
            if not self.up:
                raise httpx.ReadTimeout("still starting", request=request)
            return httpx.Response(200)
        if request.url.path == "/v1/audio/transcriptions":
            assert self.up, "audio was sent to a model that was not ready"
            assert request.headers["authorization"] == "Bearer test-key"
            body = await request.aread()
            if b'filename="warm-up.wav"' in body:
                self.warmups += 1
                if self.fail_next_piece:
                    self.fail_next_piece = False
                    return httpx.Response(500)
                return httpx.Response(200, json={"text": ""})
            self.models_seen += [m for m in ("voxtral-mini-bn-bd", "other-test-model") if m.encode() in body]
            self.transcriptions += 1
            if self.fail_next_piece:
                self.fail_next_piece = False
                return httpx.Response(500)
            self.inflight += 1
            try:
                await asyncio.sleep(self.piece_delay)
            finally:
                self.inflight -= 1
            return httpx.Response(200, json={"text": " আমি ভালো আছি "})
        return httpx.Response(404)


def audio_file(seconds=2.0, sr=16000, channels=1, fmt="WAV", subtype="PCM_16", silent=False):
    t = np.arange(int(seconds * sr)) / sr
    x = np.zeros_like(t) if silent else 0.3 * np.sin(2 * np.pi * 220 * t)
    buf = io.BytesIO()
    sf.write(buf, np.repeat(x[:, None], channels, axis=1), sr, subtype=subtype, format=fmt)
    return buf.getvalue()


def ffmpeg_convert(data: bytes, suffix: str, args):
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "in.wav", Path(d) / f"out{suffix}"
        src.write_bytes(data)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), *args, str(dst)], check=True)
        return dst.read_bytes()


async def main():
    fake, oth = Fake("voxtral.test"), Fake("other.test")
    fakes = {f.host: f for f in (fake, oth)}

    async def route(request):
        return await fakes[request.url.host].handler(request)

    asr_core = frontend_core.load_asr_core(SHARED / "asr_core.py")
    opts = dict(transport=httpx.MockTransport(route), wake_probe_timeout=0.05, wake_retry_s=0.05,
                stats_grace_s=0.3, max_upload_bytes=3_000_000, max_decoded_samples=2_000_000)
    serve_fns = {"voxtral": fake, "other": oth}
    app = frontend_core.create_app(serve_fns, "test-key", asr_core, HERE / "static", **opts)
    c = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://frontend.test")
    failures = []

    def check(name, cond):
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            failures.append(name)

    async def state(client=c, probe=False, model=None):
        q = {"probe": "1"} if probe else {}
        if model:
            q["model"] = model
        return (await client.get("/api/status", params=q)).json()["state"]

    async def until(want, client=c, tries=100, model=None):
        for _ in range(tries):
            if await state(client, model=model) == want:
                return True
            await asyncio.sleep(0.02)
        return False

    async def post(data, name="a.wav", mime="audio/wav", model=None):
        return await c.post("/api/transcribe", files={"file": (name, data, mime)},
                            data={"model": model} if model else None)

    wav = audio_file()
    check("index page is served", (await c.get("/")).status_code == 200)
    check("asleep at first", await state() == "asleep")
    check("asleep with ?probe=1, and the GPU is not touched", await state(probe=True) == "asleep" and fake.health_calls == 0)
    r = await post(b"not audio at all", "x.txt", "text/plain")
    check("a bad upload while asleep gets 415 and does not wake the GPU",
          r.status_code == 415 and fake.health_calls == 0 and fake.runners == 0)

    r = await post(wav)
    check("transcribe while asleep is refused with 503", r.status_code == 503 and fake.transcriptions == 0)
    await asyncio.sleep(0.05)
    check("...and starts waking the model", await state() == "waking" and fake.runners == 1)
    check("transcribe while waking is refused with 503", (await post(wav)).status_code == 503 and fake.transcriptions == 0)

    fake.up = True
    check("ready once vLLM answers /health", await until("ready"))
    check("...and a wake started by an upload sends no warm-up", fake.warmups == 0)
    n = fake.health_calls
    check("status while ready does not touch the GPU", await state() == "ready" and fake.health_calls == n)
    check("wake while ready does not touch the GPU",
          (await c.post("/api/wake")).json()["state"] == "ready" and fake.health_calls == n)

    r = await post(wav)
    check("transcribe when ready returns the stripped text",
          r.status_code == 200 and r.json()["text"] == "আমি ভালো আছি" and fake.transcriptions == 1)
    r = await post(audio_file(60.0), "long.wav")
    check("a 60 s file goes as 3 pieces joined with spaces",
          r.status_code == 200 and len(r.json()["pieces"]) == 3 and fake.transcriptions == 4)

    # The backend disappears right after a successful request: no cached "ready" may admit audio.
    fake.up, fake.runners = False, 0
    r = await post(wav)
    check("transcribe right after the backend went away is refused", r.status_code == 503 and fake.transcriptions == 4)
    fake.up = True
    check("ready again once it is back", await until("ready"))

    # A fresh frontend (e.g. after a redeploy) next to a running GPU container finds out by itself.
    app2 = frontend_core.create_app(serve_fns, "test-key", asr_core, HERE / "static", **opts)
    c2 = httpx.AsyncClient(transport=httpx.ASGITransport(app=app2), base_url="http://frontend2.test")
    check("a restarted frontend confirms a running container on its own", await until("ready", client=c2))
    check("...without a warm-up (status only confirms a container that exists)", fake.warmups == 0)
    await c2.aclose()

    fake.runners = 0  # autoscaler stats lagging a container that just answered
    check("a zero-runner reading right after a confirmed /health keeps ready", await state() == "ready")
    fake.up, fake.runners = False, 0
    await asyncio.sleep(0.35)  # past stats_grace_s
    check("asleep after scaling to zero", await state() == "asleep")
    check("/api/wake starts the model", (await c.post("/api/wake")).json()["state"] == "waking")
    fake.fail_next_piece = True  # the warm-up fails: the model still becomes ready
    fake.up = True
    check("ready after waking", await until("ready"))
    check("...after exactly one warm-up transcription, which failed harmlessly",
          fake.warmups == 1 and not fake.fail_next_piece and fake.transcriptions == 4)

    # Model selection: each model is woken, tracked and sent audio on its own.
    check("the other model is untouched by everything above",
          oth.health_calls == 0 and oth.runners == 0 and oth.transcriptions == 0)
    check("the other model is asleep while Voxtral is ready",
          await state(model="other") == "asleep" and await state() == "ready")
    r = await post(wav, model="other")
    check("transcribe with the other model while it sleeps is refused with 503 and wakes only it",
          r.status_code == 503 and oth.runners == 1 and oth.transcriptions == 0)
    oth.up = True
    check("the other model becomes ready on its own, with no frontend warm-up (warm_up False)",
          await until("ready", model="other") and oth.warmups == 0)
    n_v = fake.transcriptions
    r = await post(audio_file(60.0), "long.wav", model="other")
    check("its audio goes to its own server only, under its model name, in 3 pieces",
          r.status_code == 200 and r.json()["model"] == "other" and len(r.json()["pieces"]) == 3
          and oth.transcriptions == 3 and fake.transcriptions == n_v
          and oth.models_seen == ["other-test-model"] * 3)
    r = await post(wav)
    check("without a model field, audio goes to Voxtral under its model name",
          r.status_code == 200 and r.json()["model"] == "voxtral" and fake.models_seen[-1] == "voxtral-mini-bn-bd")
    oth.up, oth.runners = False, 0
    await asyncio.sleep(0.35)
    check("the other model scaling to zero leaves Voxtral ready",
          await state(model="other") == "asleep" and await state() == "ready")
    st = (await c.post("/api/wake", params={"model": "other"})).json()
    await asyncio.sleep(0.05)  # the wake loop's first /health request
    check("/api/wake?model=other wakes the other model and names it",
          st["state"] == "waking" and st["model"] == "other" and oth.runners == 1)
    oth.up = True
    check("ready again, still without a frontend warm-up", await until("ready", model="other") and oth.warmups == 0)
    check("an unknown model is refused by status, wake and transcribe",
          (await c.get("/api/status", params={"model": "x"})).status_code == 400
          and (await c.post("/api/wake", params={"model": "x"})).status_code == 400
          and (await post(wav, model="x")).status_code == 400)

    # A wake loop stops once nobody asks about its model (the page switched away, or the tab closed).
    lone = Fake("lone.test")
    fakes[lone.host] = lone
    app3 = frontend_core.create_app({"voxtral": lone}, "test-key", asr_core, HERE / "static",
                                    **{**opts, "interest_s": 0.2})
    c3 = httpx.AsyncClient(transport=httpx.ASGITransport(app=app3), base_url="http://frontend3.test")
    await c3.post("/api/wake")  # the model never comes up
    await asyncio.sleep(0.5)
    n = lone.health_calls
    await asyncio.sleep(0.3)
    check("a wake loop nobody polls stops asking /health", n > 0 and lone.health_calls == n)
    check("...and the next status for that model resumes it", await state(c3) == "waking")
    await asyncio.sleep(0.1)
    check("...asking /health again", lone.health_calls > n)
    await c3.aclose()

    # Uploads
    check("an empty upload gets 400", (await post(b"")).status_code == 400)
    check("a non-audio upload gets 415", (await post(b"not audio at all", "x.txt", "text/plain")).status_code == 415)
    check("audio shorter than 0.1 s gets 400", (await post(audio_file(0.05))).status_code == 400)
    big = b"\0" * 5_000_000  # over max_upload_bytes + 1 MiB of multipart framing
    check("an oversized upload is refused before parsing (Content-Length)", (await post(big)).status_code == 413)

    async def stream():
        boundary = b"xBOUNDARYx"
        yield b"--" + boundary + b'\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'
        for _ in range(50):
            yield b"\0" * 100_000
        yield b"\r\n--" + boundary + b"--\r\n"
    r = await c.post("/api/transcribe", content=stream(),
                     headers={"content-type": "multipart/form-data; boundary=xBOUNDARYx"})
    check("an oversized streamed upload is refused while it arrives", r.status_code == 413)

    flac = audio_file(11 * 60, sr=8000, fmt="FLAC", subtype="PCM_16", silent=True)
    r = await post(flac, "long.flac", "audio/flac")
    check(f"an 11-minute FLAC ({len(flac) // 1024} KB) is refused from its header", r.status_code == 413)
    r = await post(audio_file(30.0, sr=48000, channels=2), "hi.wav")
    check("audio over the decoded-size budget is refused from its header", r.status_code == 413)

    m3u8 = b"#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10,\nfile:///etc/hostname\n#EXT-X-ENDLIST\n"
    r = await post(m3u8, "list.m3u8", "application/vnd.apple.mpegurl")
    check("a playlist upload is refused", r.status_code == 415)
    n = fake.transcriptions
    r = await post(ffmpeg_convert(wav, ".webm", ["-c:a", "libopus"]), "rec.webm", "audio/webm")
    check("a WebM/Opus recording goes through ffmpeg and is transcribed",
          r.status_code == 200 and fake.transcriptions == n + 1)
    r = await post(ffmpeg_convert(wav, ".m4a", ["-c:a", "aac"]), "rec.m4a", "audio/mp4")
    check("an M4A/AAC recording goes through ffmpeg and is transcribed", r.status_code == 200)

    # One piece fails: the others are cancelled instead of running on.
    fake.fail_next_piece, fake.piece_delay = True, 0.5
    r = await post(audio_file(60.0), "long.wav")
    check("a failed piece returns 502 and cancels its siblings", r.status_code == 502 and fake.inflight == 0)
    fake.piece_delay = 0.0

    await c.aclose()
    print(f"{len(failures)} failed" if failures else "all passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
