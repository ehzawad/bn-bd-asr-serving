#!/usr/bin/env python
"""Transcribe audio files through a vLLM server running Voxtral Mini bn-BD: serve.sh on
your own GPU, or the Modal inference service.

    python client.py a.wav b.m4a
    python client.py recordings/ --out transcripts.jsonl --concurrency 32
    python client.py a.wav --url https://<workspace>--voxtral-mini-bn-inference-serve.us-east.modal.run \
        --api-key-file ~/.config/voxtral-bn/vllm-api-key

Chunking happens here, as in the benchmark (asr_core: 16 kHz mono, pieces of at most
25 s cut at pauses, PCM16 WAV); the server's transcription endpoint does not split
long audio for Voxtral. Each piece is one POST /v1/audio/transcriptions request with
language "bn", temperature 0 and at most 500 new tokens; requests run concurrently so
the server batches them, and each file's piece texts are joined with single spaces.
"""
import argparse
import asyncio
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ASR_CORE_SHA256 = "f894ef44ae4834fc80878f8b206e65e8321a0014807559494af0eb8880638a18"
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".mp4", ".ogg", ".opus", ".webm", ".aac", ".wma", ".aif", ".aiff"}


def load_asr_core():
    src = HERE / "asr_core.py"
    if hashlib.sha256(src.read_bytes()).hexdigest() != ASR_CORE_SHA256:
        sys.exit("asr_core.py is not the benchmark's copy")
    try:
        import torch  # noqa: F401
    except ImportError:  # asr_core needs torch only in decode(), the NeMo path, unused here
        sys.modules["torch"] = types.ModuleType("torch")
    sys.path.insert(0, str(HERE))
    import asr_core

    return asr_core


def decode(asr_core, path: Path):
    try:
        return asr_core.load_audio(str(path))
    except Exception:
        pass
    with tempfile.TemporaryDirectory() as d:  # containers libsndfile cannot read (m4a, webm, ...)
        dst = Path(d) / "decoded.wav"
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
                            "-c:a", "pcm_f32le", "-f", "wav", str(dst)], capture_output=True, timeout=600)
        if r.returncode != 0 or not dst.exists():
            raise ValueError(f"cannot decode {path}")
        return asr_core.load_audio(str(dst))


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


def wav_bytes(piece, sr) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(piece, -1.0, 1.0), sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


async def run(a, files, asr_core):
    key = os.environ.get("VLLM_API_KEY")
    if a.api_key_file:
        key = Path(a.api_key_file).expanduser().read_text().strip()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    sem = asyncio.Semaphore(a.concurrency)  # pieces in flight
    file_sem = asyncio.Semaphore(max(4, a.concurrency))  # files decoded and held in memory at once
    # follow_redirects: Modal answers a request still pending after 150 s (a cold start)
    # with a 303 to a result URL that blocks until the response is ready.
    async with httpx.AsyncClient(base_url=a.url, headers=headers, follow_redirects=True,
                                 timeout=httpx.Timeout(600, connect=30)) as client:
        async def piece(wav):
            async with sem:
                r = await client.post("/v1/audio/transcriptions", files={"file": ("piece.wav", wav, "audio/wav")},
                                      data={"model": a.model, "language": "bn", "temperature": "0",
                                            "max_completion_tokens": str(a.max_tokens), "response_format": "json"})
                r.raise_for_status()
                return r.json()["text"].strip()

        async def one(path):
            async with file_sem:
                return await one_file(path)

        async def one_file(path):
            try:
                audio, note = await asyncio.to_thread(decode, asr_core, path)
            except Exception as e:  # unreadable file, ffmpeg missing or timed out: record it, keep going
                return {"file": str(path), "error": f"{type(e).__name__}: {e}"}
            pieces, forced = asr_core.chunks(audio)
            sr = asr_core.TARGET_SR
            try:
                texts = await gather_or_cancel(*(piece(wav_bytes(p, sr)) for p in pieces))
            except httpx.HTTPError as e:
                return {"file": str(path), "error": f"{type(e).__name__}: {e}"}
            return {"file": str(path), "text": " ".join(t for t in texts if t), "pieces": texts,
                    "duration_s": round(len(audio) / sr, 3), "forced_cuts": forced, "note": note}

        return await asyncio.gather(*(one(p) for p in files))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", help="audio files or directories (searched recursively)")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key-file", help="file holding the server's API key (else $VLLM_API_KEY, if set)")
    ap.add_argument("--model", default="voxtral-mini-bn-bd", help="the server's --served-model-name")
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=16, help="pieces in flight at once")
    ap.add_argument("--out", default="-", help="JSONL output path, or - for stdout (default)")
    a = ap.parse_args()
    if a.concurrency < 1 or a.max_tokens < 1:
        ap.error("--concurrency and --max-tokens must be positive")

    files = []
    for p in map(Path, a.inputs):
        files += sorted(q for q in p.rglob("*") if q.suffix.lower() in AUDIO_EXTS) if p.is_dir() else [p]
    asr_core = load_asr_core()
    t = time.perf_counter()
    results = asyncio.run(run(a, files, asr_core))
    wall = time.perf_counter() - t
    out = sys.stdout if a.out == "-" else open(a.out, "w", encoding="utf-8")
    for r in results:
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
    if out is not sys.stdout:
        out.close()
    failed = sum("error" in r for r in results)
    audio_s = sum(r.get("duration_s", 0) for r in results)
    print(f"{len(results) - failed} files, {audio_s / 60:.1f} min of audio in {wall:.1f} s; {failed} failed",
          file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
