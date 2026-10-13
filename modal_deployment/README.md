# Modal deployment

**Live:** <https://ehzawad-56775--voxtral-bn-asr.us-east.modal.run> (the frontend app below; the page wakes the inference app when it is opened).

Two Modal apps, deployed and scaled independently:

```
 browser ──HTTPS──▶  voxtral-mini-bn-frontend  (CPU)      ──HTTPS + API key──▶  voxtral-mini-bn-inference  (1× L4)
                     page, readiness, chunking                                   vllm serve, OpenAI-compatible
                     frontend_app.py / frontend_core.py                          inference_app.py
```

- **Inference** (`inference_app.py`) runs `vllm serve` for the model on one L4:
  - Settings: eager mode, `max_model_len` 2048, up to 32 concurrent pieces, `/v1/*` guarded by `VLLM_API_KEY`.
  - Its `/health` returns 200 only once the engine has fully loaded.
  - It scales to zero 10 minutes after its last request, and `max_containers=1` caps spend at one L4.
- **Frontend** (`frontend_app.py` → `frontend_core.py`, `static/index.html`) is a small CPU container that never loads a model:
  - It decodes and chunks uploads with the benchmark's `asr_core`, sends each ≤ 25 s piece to vLLM's `/v1/audio/transcriptions` and joins the texts.
  - It scales to zero 5 minutes after its last request.

## Readiness: nothing is accepted until vLLM is ready

1. **On load:** opening the page wakes the inference service.
2. **While waking:** Record and Upload stay locked behind a "waking the model" notice. A cold start takes about 1–3 minutes.
3. **Unlocking:** the controls unlock only after vLLM's `/health` has answered 200.
4. **Before each use:** right before recording or uploading, the page asks vLLM directly (`/api/status?probe=1`).
5. **On the server:** `/api/transcribe` decodes and checks the upload first, so a bad file never touches the GPU. It then asks vLLM again right before sending any audio, and refuses with 503 (starting the wake) if vLLM does not answer.
6. **During a recording:** the page keeps the GPU warm while someone is recording, so a long recording never meets a sleeping model. If the permission prompt stays open while the page pauses or the model goes away, the recording does not start.

## Cost: nothing runs while nobody uses it

- **GPU:** the L4 container stops 10 minutes after its last request.
  - Status polls read Modal's autoscaler stats (`Function.get_current_stats()`), which never reach the GPU container, so an open tab doesn't keep it awake.
- **Page:** it stops polling while hidden and after 5 minutes without a tap or key press. It then shows "Paused… tap anywhere to continue".
- **Frontend:** its CPU container stops 5 minutes later, so nothing runs 10 minutes after the last tap.
- **Broken model:** if the model fails to start, the frontend gives up after 20 minutes and lets it scale down, instead of retrying forever.
- **Billing:** Modal bills only while containers run. L4 time costs $0.000222/s (about $0.80/h).
  - The cached weights (about 9.4 GB in a Volume) fall within the 1 TiB/month of free Volume storage.

## Deploy

```bash
# once: secrets (values never printed)
modal secret create huggingface-voxtral --from-json hf.json        # {"HF_TOKEN": "..."}, read access to the private model
modal secret create voxtral-bn-vllm-key --from-json key.json       # {"VLLM_API_KEY": "<random>"}, shared by both apps

# once: cache the Mistral-format weights (consolidated.safetensors, params.json, tekken.json) in a Volume
modal run modal_deployment/inference_app.py::download_weights

modal deploy modal_deployment/inference_app.py
modal deploy modal_deployment/frontend_app.py   # prints the page URL: https://<workspace>--voxtral-bn-asr.<region>.modal.run
```

Redeploying the frontend with `--strategy recreate` makes the new version serve immediately.

## Use it from code

The frontend's API needs no key:

```bash
curl -F file=@speech.m4a https://<workspace>--voxtral-bn-asr.<region>.modal.run/api/transcribe
# {"text": "...", "pieces": [...], "audio_seconds": 16.2, ...}   (503 while the model is not ready)
```

The inference service speaks vLLM's OpenAI-compatible API and needs the key. Use `../vllm_inference/client.py --url <inference URL> --api-key-file <file>`, which also does the benchmark's chunking.

Accepted input: WAV, FLAC, MP3, OGG/Opus, M4A/AAC, WebM, and video files with an audio track, up to 10 minutes and 100 MB. Language is fixed to Bangla.
- **Size limits:** oversized bodies are refused while they arrive. The duration and decoded size are checked from the file header before decoding, so a tiny compressed file can't expand into gigabytes.
- **Playlists:** ffmpeg only opens audio and video containers, so playlists and concat lists are refused.

## Test

`python modal_deployment/test_frontend.py` drives the real web app against a fake autoscaler and a fake vLLM. It needs fastapi, httpx, python-multipart, numpy, soundfile and librosa, but no Modal and no GPU. It runs 30 checks:
- every readiness transition: asleep, waking, ready, a backend that vanishes right after a success, a restarted frontend, stats lag, scale-down;
- that no audio reaches vLLM early and that bad uploads never wake the GPU;
- upload limits, decompression bombs, playlists, WebM and M4A decoding;
- cancellation of sibling requests when one piece fails.
