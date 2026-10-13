#!/usr/bin/env python
"""Offline check of the frontend's readiness gating and upload handling, against a fake Modal
autoscaler and a fake vLLM (no Modal, no GPU). Needs: fastapi, httpx, python-multipart,
numpy, soundfile, librosa, and ffmpeg/ffprobe on PATH.

    python modal_deployment/test_frontend.py
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
sys.path.insert(0, str(HERE))
import frontend_core  # noqa: E402


class Fake:
    """A Modal web function (autoscaler stats) plus the vLLM server behind it."""

    def __init__(self):
        self.runners = 0  # containers, starting or up
        self.up = False  # the engine has loaded and /health answers 200
        self.health_calls = 0
        self.transcriptions = 0
        self.fail_next_piece = False
        self.piece_delay = 0.0
        self.inflight = 0
        self.get_web_url = types.SimpleNamespace(aio=self._url)
        self.get_current_stats = types.SimpleNamespace(aio=self._stats)

    async def _url(self):
        return "http://inference.test"

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
    fake = Fake()
    asr_core = frontend_core.load_asr_core(HERE / "asr_core.py")
    opts = dict(transport=httpx.MockTransport(fake.handler), wake_probe_timeout=0.05, wake_retry_s=0.05,
                stats_grace_s=0.3, max_upload_bytes=3_000_000, max_decoded_samples=2_000_000)
    app = frontend_core.create_app(fake, "test-key", asr_core, HERE / "static", **opts)
    c = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://frontend.test")
    failures = []

    def check(name, cond):
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            failures.append(name)

    async def state(client=c, probe=False):
        return (await client.get("/api/status" + ("?probe=1" if probe else ""))).json()["state"]

    async def until(want, client=c, tries=100):
        for _ in range(tries):
            if await state(client) == want:
                return True
            await asyncio.sleep(0.02)
        return False

    async def post(data, name="a.wav", mime="audio/wav"):
        return await c.post("/api/transcribe", files={"file": (name, data, mime)})

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
    app2 = frontend_core.create_app(fake, "test-key", asr_core, HERE / "static", **opts)
    c2 = httpx.AsyncClient(transport=httpx.ASGITransport(app=app2), base_url="http://frontend2.test")
    check("a restarted frontend confirms a running container on its own", await until("ready", client=c2))
    await c2.aclose()

    fake.runners = 0  # autoscaler stats lagging a container that just answered
    check("a zero-runner reading right after a confirmed /health keeps ready", await state() == "ready")
    fake.up, fake.runners = False, 0
    await asyncio.sleep(0.35)  # past stats_grace_s
    check("asleep after scaling to zero", await state() == "asleep")
    check("/api/wake starts the model", (await c.post("/api/wake")).json()["state"] == "waking")
    fake.up = True
    check("ready after waking", await until("ready"))

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
