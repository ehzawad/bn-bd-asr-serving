# Fast local inference with vLLM

These tools run on any CUDA GPU with about 16 GB or more of memory. The bf16 weights take 9.4 GB; the rest is KV cache and activations. Install:

```bash
pip install -r requirements.txt        # vllm[audio] 0.31.0 (torch 2.13, CUDA 13.0), mistral-common, librosa, soundfile, httpx
hf auth login                          # the model repo is private; or export HF_TOKEN=...
```

`ffmpeg` on `PATH` lets the tools read M4A, WebM and other containers that libsndfile can't.

## `transcribe.py`: batch transcription, fastest

```bash
python transcribe.py recordings/ --out transcripts.jsonl     # directories are searched recursively
python transcribe.py a.wav b.m4a                             # JSONL to stdout
python transcribe.py --list files.txt --out out.jsonl --eager
```

- **How it decodes:** pieces from many files go through vLLM's continuous batching, with CUDA graphs on by default. `--eager` starts faster but decodes more slowly, and matches the benchmark's engine mode.
- **Speed:** on an A100-40GB, the 920 FLEURS test clips (3.43 h of audio, 940 pieces) took 31 s of generation, 395× real time. Audio prep took 9 s and engine load 56 s.
- **Output:** one JSON object per file, in input order, with these fields:
  - `text` and `pieces`
  - `duration_s` and `forced_cuts`
  - `hit_token_cap`, set when a piece ran to the 500-token cap (usually a repetition loop)
- **Errors:** unreadable files get an `error` field, and the exit status is 1 if any failed.

## `serve.sh` + `client.py`: an OpenAI-compatible server

```bash
export VLLM_API_KEY="$(openssl rand -hex 24)"               # shared by the server and the client below
./serve.sh &                                                # http://127.0.0.1:8000, same settings as the Modal service
python client.py a.wav recordings/ --out out.jsonl          # reads $VLLM_API_KEY; --api-key-file also works
python client.py a.wav --url https://<workspace>--voxtral-mini-bn-inference-serve.<region>.modal.run \
    --api-key-file ~/.config/voxtral-bn/vllm-api-key        # the Modal inference service
```

Long audio has to be chunked by the client. vLLM's transcription endpoint doesn't split audio for Voxtral, so it would encode the whole file as one prompt. `client.py` cuts ≤ 25 s pieces at pauses, exactly as the benchmark did, and sends them concurrently so the server batches them.

The server can also be called directly with any OpenAI-style client: `POST /v1/audio/transcriptions` with `model=voxtral-mini-bn-bd`, `language=bn`, `temperature=0` and `max_completion_tokens=500`. Each request should carry one piece of at most 25 s.

## Parity with the benchmark

FLEURS test, against the benchmark's transcripts:
- `transcribe.py` matched 875/920 transcripts, with WER 14.72% against the benchmark's 14.73%.
- `client.py` against `serve.sh` matched 283 of the first 300 transcripts.

Engine settings (CUDA graphs, batch composition, GPU type) change a few transcripts in both directions, but not the aggregate WER.
