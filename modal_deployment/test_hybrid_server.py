#!/usr/bin/env python
"""Offline check of the hybrid FastConformer's inference server (hybrid_core.py) with a stand-in
for the NeMo model (no Modal, no NeMo, no GPU). Needs: fastapi, httpx, python-multipart, numpy,
soundfile.

    python modal_deployment/test_hybrid_server.py
"""
import asyncio
import io
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hybrid_core  # noqa: E402


class Hyp:
    def __init__(self, text):
        self.text = text


class FakeModel:
    """Records what NeMo's transcribe() would be given; answers like NeMo (Hypothesis objects)."""

    def __init__(self):
        self.calls, self.seen, self.active, self.max_active = [], [], 0, 0
        self.lock = threading.Lock()

    def transcribe(self, paths, batch_size=1, verbose=False):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.calls.append((list(paths), batch_size, verbose))
            self.seen.append(Path(paths[0]).read_bytes())
            time.sleep(0.05)
            return [Hyp("  আমি ভালো আছি ")]
        finally:
            with self.lock:
                self.active -= 1


def wav(seconds=2.0, sr=16000, channels=1, subtype="PCM_16", fmt="WAV"):
    x = 0.3 * np.sin(2 * np.pi * 220 * np.arange(int(seconds * sr)) / sr)
    buf = io.BytesIO()
    sf.write(buf, np.repeat(x[:, None], channels, axis=1), sr, subtype=subtype, format=fmt)
    return buf.getvalue()


async def main():
    model = FakeModel()
    app = hybrid_core.create_app(model, "test-key")
    c = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://hybrid.test")
    key = {"Authorization": "Bearer test-key"}
    failures = []

    def check(name, cond):
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            failures.append(name)

    async def post(data, headers=key, model=hybrid_core.SERVED_MODEL_NAME):
        return await c.post("/v1/audio/transcriptions", headers=headers,
                            files={"file": ("piece.wav", data, "audio/wav")}, data={"model": model})

    check("a warm-up decode ran before the app was built", len(model.calls) == 1)
    check("/health answers 200", (await c.get("/health")).status_code == 200)

    piece = wav()
    r = await post(piece)
    check("a 16 kHz mono PCM16 piece is transcribed, text stripped as asr_core.decode does",
          r.status_code == 200 and r.json() == {"text": "আমি ভালো আছি"})
    check("the model gets the piece's bytes unchanged, one path, batch 1, not verbose",
          model.seen[-1] == piece and len(model.calls[-1][0]) == 1 and model.calls[-1][1:] == (1, False))

    check("no key gets 401", (await post(piece, headers={})).status_code == 401)
    check("a wrong key gets 401", (await post(piece, headers={"Authorization": "Bearer nope"})).status_code == 401)
    check("another model name gets 404", (await post(piece, model="voxtral-mini-bn-bd")).status_code == 404)
    n = len(model.calls)
    check("a non-WAV piece gets 400", (await post(b"not audio")).status_code == 400)
    check("a 48 kHz piece gets 400", (await post(wav(sr=48000))).status_code == 400)
    check("a stereo piece gets 400", (await post(wav(channels=2))).status_code == 400)
    check("a float WAV piece gets 400", (await post(wav(subtype="FLOAT"))).status_code == 400)
    check("a FLAC piece gets 400", (await post(wav(fmt="FLAC"))).status_code == 400)
    check("a piece over 30 s gets 413", (await post(wav(31.0))).status_code == 413)
    check("an oversized body gets 413", (await post(b"\0" * (hybrid_core.MAX_PIECE_BYTES + 10))).status_code == 413)
    check("no rejected piece reached the model", len(model.calls) == n)

    def streamed(chunks, counter):
        async def gen():
            yield b'--xB\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'
            for _ in range(chunks):
                counter.append(1)
                yield b"\0" * 100_000
            yield b"\r\n--xB--\r\n"
        return gen()

    pulled = []
    r = await c.post("/v1/audio/transcriptions", content=streamed(80, pulled),
                     headers={"content-type": "multipart/form-data; boundary=xB"})
    check("an 8 MB body without a key is refused before it is read", r.status_code == 401 and len(pulled) == 0)
    pulled = []
    r = await c.post("/v1/audio/transcriptions", content=streamed(80, pulled),
                     headers={**key, "content-type": "multipart/form-data; boundary=xB"})
    check("an 8 MB body with the key is refused while it arrives", r.status_code == 413 and len(pulled) < 30)
    r = await c.post("/v1/audio/transcriptions", content=b"x" * 10,
                     headers={**key, "content-type": "multipart/form-data; boundary=xB",
                              "content-length": str(hybrid_core.MAX_BODY_BYTES + 1)})
    check("a declared oversized body is refused from its Content-Length", r.status_code == 413)
    check("none of these reached the model", len(model.calls) == n)

    rs = await asyncio.gather(*(post(piece) for _ in range(6)))
    check("concurrent pieces all succeed, decoded one at a time",
          all(r.status_code == 200 for r in rs) and model.max_active == 1)

    await c.aclose()
    print(f"{len(failures)} failed" if failures else "all passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
