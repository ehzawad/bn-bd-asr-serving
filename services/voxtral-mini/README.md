# Voxtral Mini inference service (FP8)

[`ehzawad/Voxtral-Mini-3B-2507-bn-BD`](https://huggingface.co/ehzawad/Voxtral-Mini-3B-2507-bn-BD) at its [`fp8`](https://huggingface.co/ehzawad/Voxtral-Mini-3B-2507-bn-BD/tree/fp8) branch (commit `287697acc3ab`), served with vLLM 0.31. The text decoder is FP8: E4M3 weights with one scale per output channel, and activations quantized per token at run time. The audio encoder stays BF16. Its WER is within 0.5 points of the BF16 snapshot `d1c7aeb0cd18` on ten of the benchmark's eleven sets (see the model card). Set `MODEL_REVISION` (or `REVISION` / `--revision`) to `d1c7aeb0cd18f8a7a057bf756df5ee2ccb7aebcd` to serve BF16 again. The folder holds the Modal service behind the page (`modal_app.py`) and the same server and a batch tool for any CUDA GPU.

## On Modal: `modal_app.py`

Modal app `voxtral-mini-bn-inference`, at `https://<workspace>--voxtral-mini-bn-inference-serve.<region>.modal.run`. It runs `vllm serve` on one L4:
- **Settings:** torch.compile with CUDA graphs (`FAST_BOOT = False`), `max_model_len` 2048, up to 32 concurrent pieces.
- **Cold start from a GPU memory snapshot.** The class `Voxtral` boots once: vLLM loads, compiles, captures CUDA graphs and answers two warm-up pieces. vLLM then goes to sleep level 1, which moves the weights to CPU RAM and releases the KV cache, and Modal snapshots CPU and GPU memory. Every later cold start restores that snapshot and wakes vLLM in ~1.3 s. A full boot takes ~5 min, and the first 2–3 wakes after each deploy are full boots while Modal makes the snapshots.
- **nginx in front.** vLLM listens on localhost. nginx on port 8000 passes only `/v1/*` and `/health`, because the snapshot needs vLLM's dev-mode `/sleep` and `/wake_up` routes, which the API key does not guard.
- **Auth:** `/v1/*` is guarded by `VLLM_API_KEY`. `/health` is open and returns 200 only once the engine has fully loaded.
- **Scaling:** it scales to zero 10 minutes after its last request, and `max_containers=1` caps spend at one L4.
- **Speed (L4, from inside Modal, 2026-10-19):** about 0.5 s for a 5 s piece and 1.3–1.6 s for a 16 s piece. A cold start takes about half a minute (snapshot restore).
  - FP8 against BF16, same L4, same clips, median of 6 requests each (16 s / 7 s / 5 s pieces):

    | Mode | BF16 | FP8 | FP8 speedup |
    |---|---|---|---|
    | CUDA graphs (served) | 2.43 / 1.12 / 0.80 s | 1.56 / 0.72 / 0.51 s | 1.56× |
    | Eager (`FAST_BOOT = True`) | 2.49 / 1.08 / 0.75 s | 2.78 / 1.28 / 0.82 s | 0.88× (slower) |

    16 concurrent 16 s pieces took 2.20 s with FP8 and 3.17 s with BF16, both with CUDA graphs. With CUDA graphs, all five FP8 transcripts matched BF16 exactly.
  - Eager mode is the only case where FP8 is slower: the per-token activation quantization fuses into the FP8 matmuls only when compiled. So the service compiles, even though boot takes longer. Most of the boot is a ~2 minute stretch before CUDA-graph capture. The torch.compile artifacts are cached in the `voxtral-bn-vllm-cache` Volume, and loading them takes about 35 s.
  - A fresh vLLM spends about 1.1 s more on its first request, so the frontend sends a warm-up piece before it unlocks the page.
- **Billing:** Modal bills only while the container runs. L4 time costs $0.000222/s (about $0.80/h). The cached weights (about 6.2 GB, in a Volume) fall within the 1 TiB/month of free Volume storage.

```bash
# once: secrets (values never printed)
modal secret create huggingface-voxtral --from-json hf.json        # {"HF_TOKEN": "..."}, read access to the private model
modal secret create voxtral-bn-vllm-key --from-json key.json       # {"VLLM_API_KEY": "<random>"}, shared with the frontend

modal run services/voxtral-mini/modal_app.py::download_weights     # once: Mistral-format files into a Volume
modal deploy services/voxtral-mini/modal_app.py
```

The service follows the contract in the [top-level README](../../README.md#adding-a-model), so the frontend can call it like any other model.

## On any CUDA GPU

These tools run on any CUDA GPU with about 12 GB or more of memory. The FP8 weights take 6.1 GB; the rest is KV cache and activations. FP8 kernels need Ada, Hopper or Blackwell (L4, RTX 40xx, H100 and newer). On an older GPU such as an A100, vLLM falls back to weight-only FP8, which is untested here; to run BF16 there instead, pass the BF16 revision above (9.4 GB). Install:

```bash
pip install -r requirements.txt        # vllm[audio] 0.31.0 (torch 2.13, CUDA 13.0), mistral-common, librosa, soundfile, httpx
hf auth login                          # the model repo is private; or export HF_TOKEN=...
```

`ffmpeg` on `PATH` lets the tools read M4A, WebM and other containers that libsndfile can't.

`client.py` and `transcribe.py` load the benchmark's chunker from `../../shared/asr_core.py`, so run them from a checkout of this repo.

### `transcribe.py`: batch transcription, fastest

```bash
python transcribe.py recordings/ --out transcripts.jsonl     # directories are searched recursively
python transcribe.py a.wav b.m4a                             # JSONL to stdout
python transcribe.py --list files.txt --out out.jsonl --eager
```

- **How it decodes:** pieces from many files go through vLLM's continuous batching, with CUDA graphs on by default. The benchmark's batch-1 run used CUDA graphs too. `--eager` starts faster but decodes more slowly.
- **Speed (BF16):** on an A100-40GB, the 920 FLEURS test clips (3.43 h of audio, 940 pieces) took 31 s of generation, 395× real time. Audio prep took 9 s and engine load 56 s.
- **Output:** one JSON object per file, in input order, with these fields:
  - `text` and `pieces`
  - `duration_s` and `forced_cuts`
  - `hit_token_cap`, set when a piece ran to the 500-token cap (usually a repetition loop)
- **Errors:** unreadable files get an `error` field, and the exit status is 1 if any failed.

### `serve.sh` + `client.py`: an OpenAI-compatible server

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

Measured with the BF16 snapshot. FLEURS test, against the benchmark's earlier batched decode:
- `transcribe.py` matched 875/920 transcripts, with WER 14.72% against the benchmark's 14.73%.
- `client.py` against `serve.sh` matched 283 of the first 300 transcripts.
- The Modal service, through the page's API on the L4, matched 853/920, with WER 14.69%.
- The benchmark's own batch-1 run matched 860/920, with the same WER.

Engine settings (CUDA graphs, batch composition, GPU type) change a few transcripts in both directions, but not the aggregate WER.
