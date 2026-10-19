#!/usr/bin/env python
"""Fast offline Bangla transcription with Voxtral Mini bn-BD on a local CUDA GPU.

    python transcribe.py recordings/ --out transcripts.jsonl
    python transcribe.py a.wav b.m4a --out -
    python transcribe.py --list files.txt --out transcripts.jsonl

Each file goes through the benchmark's audio path (asr_core: 16 kHz mono, pieces of
at most 25 s cut at pauses, PCM16 staging), and the pieces of many files are decoded
together in vLLM's continuous batches: greedy, language "bn", at most 500 new tokens
per piece. Each file's piece texts are joined with single spaces, as the benchmark
joined them. Prompts are built exactly as vLLM's transcription endpoint builds them
for Voxtral, so this tool, serve.sh + client.py and the Modal service all send the
model the same input.

CUDA graphs are on by default (faster decoding after a slower start), as in the
benchmark's batch-1 run; --eager starts faster. One JSON object per input file is
written, in input order; a piece that reached the token cap usually means the model
fell into a repetition loop and is flagged.
"""
import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

SHARED = Path(__file__).resolve().parents[2] / "shared"  # the repo's one asr_core.py
MODEL = "ehzawad/Voxtral-Mini-3B-2507-bn-BD"
REVISION = "287697acc3abeda13de40ecae073e07b67947b9c"  # branch fp8: FP8 decoder (FP8 kernels on Ada, Hopper, Blackwell)
# asr_core.py is copied verbatim from github.com/ehzawad/bengali-asr-benchmark@5cccd1a
ASR_CORE_SHA256 = "f894ef44ae4834fc80878f8b206e65e8321a0014807559494af0eb8880638a18"
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".mp4", ".ogg", ".opus", ".webm", ".aac", ".wma", ".aif", ".aiff"}


def load_asr_core():
    src = SHARED / "asr_core.py"
    if hashlib.sha256(src.read_bytes()).hexdigest() != ASR_CORE_SHA256:
        sys.exit("asr_core.py is not the benchmark's copy")
    try:
        import torch  # noqa: F401  (vLLM installs it; asr_core imports it for its NeMo path)
    except ImportError:
        sys.modules["torch"] = types.ModuleType("torch")
    sys.path.insert(0, str(SHARED))
    import asr_core

    return asr_core


def decode(asr_core, path: Path):
    """16 kHz mono float32. Containers libsndfile cannot read go through ffmpeg first
    (to a float WAV at the source rate and channel count), so asr_core still does the
    downmix and resampling."""
    try:
        return asr_core.load_audio(str(path))
    except Exception:
        pass
    with tempfile.TemporaryDirectory() as d:
        dst = Path(d) / "decoded.wav"
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
                            "-c:a", "pcm_f32le", "-f", "wav", str(dst)], capture_output=True, timeout=600)
        if r.returncode != 0 or not dst.exists():
            raise ValueError(f"cannot decode {path}: {r.stderr.decode(errors='replace').strip()[:200]}")
        return asr_core.load_audio(str(dst))


def staged(piece, sr):
    """asr_core.decode's staging (clip, PCM16 WAV), read back as float32."""
    buf = io.BytesIO()
    sf.write(buf, np.clip(piece, -1.0, 1.0), sr, subtype="PCM_16", format="WAV")
    buf.seek(0)
    wav, _ = sf.read(buf, dtype="float32")
    return wav


def collect(args):
    paths = [Path(p) for p in args.inputs]
    if args.list:
        paths += [Path(line.strip()) for line in Path(args.list).read_text().splitlines() if line.strip()]
    files = []
    for p in paths:
        if p.is_dir():
            files += sorted(q for q in p.rglob("*") if q.suffix.lower() in AUDIO_EXTS)
        else:
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="*", help="audio files or directories (searched recursively)")
    ap.add_argument("--list", help="text file with one audio path per line")
    ap.add_argument("--out", default="-", help="JSONL output path, or - for stdout (default)")
    ap.add_argument("--model", default=MODEL, help="HF repo id or a local Mistral-format directory")
    ap.add_argument("--revision", default=REVISION, help="HF revision (ignored for a local directory)")
    ap.add_argument("--eager", action="store_true", help="no torch.compile / CUDA graphs: faster start, slower decode")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-num-batched-tokens", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-tokens", type=int, default=500, help="per piece; the benchmark used 500")
    ap.add_argument("--files-per-batch", type=int, default=500,
                    help="files decoded per generate call; bounds host memory (prompts hold 30 s-padded audio)")
    ap.add_argument("--workers", type=int, default=8, help="threads for audio decoding")
    a = ap.parse_args()
    for name in ("files_per_batch", "workers", "max_tokens", "max_num_seqs", "max_num_batched_tokens"):
        if getattr(a, name) < 1:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    os.environ.setdefault("VLLM_LOGGING_STREAM", "ext://sys.stderr")  # stdout carries only the JSONL

    files = collect(a)
    if not files:
        ap.error("no audio files given")
    asr_core = load_asr_core()
    sr = asr_core.TARGET_SR

    from mistral_common.protocol.transcription.request import TranscriptionRequest
    from mistral_common.tokens.tokenizers.audio import Audio
    from vllm import LLM, SamplingParams

    local = Path(a.model).is_dir()
    t0 = time.perf_counter()
    llm = LLM(model=a.model, revision=None if local else a.revision,
              tokenizer_revision=None if local else a.revision,
              tokenizer_mode="mistral", config_format="mistral", load_format="mistral",
              max_model_len=2048, max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_num_batched_tokens,
              gpu_memory_utilization=a.gpu_memory_utilization, limit_mm_per_prompt={"audio": 1},
              enforce_eager=a.eager, seed=0)
    load_s = time.perf_counter() - t0
    instruct = llm.get_tokenizer().instruct
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)

    def load(path):  # threads: decoding and resampling only
        try:
            audio, note = decode(asr_core, path)
        except Exception as e:  # report and keep going
            return {"file": str(path), "error": str(e)}
        pieces, forced = asr_core.chunks(audio)
        return {"file": str(path), "duration_s": round(len(audio) / sr, 3), "forced_cuts": forced,
                "note": note, "wavs": [staged(p, sr) for p in pieces]}

    def prompt(wav):  # main thread: the tokenizer is not documented as thread-safe
        req = TranscriptionRequest(model=a.model, audio=Audio(wav, sr, format="wav").to_base64("wav"),
                                   language="bn")
        tok = instruct.encode_transcription(req)
        return {"prompt_token_ids": tok.tokens,
                "multi_modal_data": {"audio": [(x.audio_array, sr) for x in tok.audios]}}

    out = sys.stdout if a.out == "-" else open(a.out, "w", encoding="utf-8")
    audio_s = gen_s = prep_s = 0.0
    n_pieces = n_capped = n_failed = 0
    with ThreadPoolExecutor(a.workers) as pool:
        for b in range(0, len(files), a.files_per_batch):
            t = time.perf_counter()
            batch = list(pool.map(load, files[b:b + a.files_per_batch]))
            prompts = [prompt(w) for r in batch if "wavs" in r for w in r["wavs"]]
            prep_s += time.perf_counter() - t
            t = time.perf_counter()
            outs = llm.generate(prompts, sp, use_tqdm=sys.stderr.isatty()) if prompts else []
            gen_s += time.perf_counter() - t
            k = 0
            for r in batch:
                if "error" in r:
                    n_failed += 1
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    continue
                res = outs[k:k + len(r.pop("wavs"))]
                k += len(res)
                texts = [o.outputs[0].text.strip() for o in res]
                capped = [o.outputs[0].finish_reason == "length" for o in res]
                r.update(text=" ".join(x for x in texts if x), pieces=texts, hit_token_cap=any(capped))
                audio_s += r["duration_s"]
                n_pieces += len(texts)
                n_capped += sum(capped)
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.flush()
    if out is not sys.stdout:
        out.close()
    print(f"{len(files) - n_failed} files ({audio_s / 3600:.2f} h of audio, {n_pieces} pieces) in "
          f"{gen_s:.1f} s of generation = {audio_s / max(gen_s, 1e-9):.0f}x real time; audio prep {prep_s:.1f} s, "
          f"engine load {load_s:.0f} s; {n_failed} unreadable, {n_capped} pieces hit the token cap",
          file=sys.stderr)
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
