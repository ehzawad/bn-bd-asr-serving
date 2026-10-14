# Modal deployment

**Live:** <https://ehzawad-56775--voxtral-bn-asr.us-east.modal.run>. This is the frontend app below. The page has a model picker: **Voxtral Mini 3B bn-BD** or the **FastConformer hybrid bn-BD**, which is `?model=hybrid` in the URL. The page wakes only the selected model.

Three Modal apps, deployed and scaled independently:

```
                                                      ┌─HTTPS + API key─▶  voxtral-mini-bn-inference         (1× L4)
 browser ──HTTPS──▶  voxtral-mini-bn-frontend  (CPU) ─┤                    vllm serve, OpenAI-compatible
                     page, model picker, readiness,   │                    inference_app.py
                     chunking                         └─HTTPS + API key─▶  fastconformer-hybrid-bn-inference (1× L4)
                     frontend_app.py / frontend_core.py                     NeMo, the same two routes
                                                                            hybrid_inference_app.py / hybrid_core.py
```

- **Inference** (`inference_app.py`) runs `vllm serve` for the model on one L4:
  - Settings: eager mode, `max_model_len` 2048, up to 32 concurrent pieces, `/v1/*` guarded by `VLLM_API_KEY`.
  - Its `/health` returns 200 only once the engine has fully loaded.
  - It scales to zero 10 minutes after its last request, and `max_containers=1` caps spend at one L4.
- **Hybrid inference** (`hybrid_inference_app.py` → `hybrid_core.py`) runs [`ehzawad/fastconformer-hybrid-rnnt-ctc-large-bn-BD`](https://huggingface.co/ehzawad/fastconformer-hybrid-rnnt-ctc-large-bn-BD) on one L4. It decodes the way the benchmark ran it:
  - the benchmarked `.nemo` snapshot (`47a7c3d5`), with torch 2.11.0 (CUDA 12.8) and `nemo_toolkit[asr]` 2.7.3, the benchmark's pins;
  - the model's default RNN-T decoder (greedy), batch 1, one piece at a time;
  - each piece's WAV bytes passed unchanged to `model.transcribe`, as `asr_core.decode` does.
  - It answers the same two routes as vLLM, `/health` and `/v1/audio/transcriptions`, behind the same API key, so the frontend treats both models alike.
  - The model loads and decodes one warm-up piece before any request is routed to the container.
  - Requests without the key, and bodies over about 2 MB, are refused before the upload is parsed. Only 16 kHz mono PCM16 WAV pieces of at most 30 s reach the model.
  - It scales to zero 10 minutes after its last request, and `max_containers=1` caps spend at one L4.
- **Frontend** (`frontend_app.py` → `frontend_core.py`, `static/index.html`) is a small CPU container that never loads a model:
  - It decodes and chunks uploads with the benchmark's `asr_core`, sends each ≤ 25 s piece to the selected model's `/v1/audio/transcriptions` and joins the texts.
  - It tracks each model's readiness separately; every endpoint takes `?model=voxtral|hybrid`, and the transcribe form takes a `model` field. The default is `voxtral`.
  - It scales to zero 5 minutes after its last request.

### Speed (measured on the L4s, from inside Modal)

| | 3 s piece | 16 s piece | Cold start |
|---|---:|---:|---:|
| FastConformer hybrid | 0.50 s | 0.68 s | about 35–90 s |
| Voxtral Mini (vLLM, eager) | 0.82 s | 2.34 s | about 1–3 min |

- **Hybrid:** the model itself takes about 0.09 s per 16 s piece on the L4, which matches the A100 benchmark. Most of the time above is Modal's web-endpoint round trip, about 0.2–0.3 s per request, plus the upload.
- **Voxtral:** a fresh vLLM spends about 1.1 s more on its first request. The frontend therefore sends a warm-up piece before it unlocks the page.

## Readiness: nothing is accepted until the selected model is ready

1. **On load:** opening the page wakes the selected model's inference service, and only that one. Switching models wakes the newly selected one; the other sleeps after its idle minutes. The picker is locked while recording or transcribing.
2. **While waking:** Record and Upload stay locked behind a "waking the model" notice. A cold start takes about 1–3 minutes for Voxtral and about a minute for the hybrid.
3. **Unlocking:** the controls unlock only after the model's `/health` has answered 200.
   - For Voxtral Mini, the page also waits for one short warm-up transcription.
   - The hybrid's server warms itself up before it answers `/health`.
4. **Before each use:** right before recording or uploading, the page asks the model directly (`/api/status?model=…&probe=1`).
5. **On the server:** `/api/transcribe` decodes and checks the upload first, so a bad file never touches the GPU. It then asks the model again right before sending any audio, and refuses with 503 (starting the wake) if the model does not answer.
6. **Comparing models:** after a transcript, switching models offers "Transcribe with …" for the same audio once the other model is ready.
7. **During a recording:** the page keeps the GPU warm while someone is recording, so a long recording never meets a sleeping model. If the permission prompt stays open while the page pauses or the model goes away, the recording does not start.

## Cost: nothing runs while nobody uses it

- **GPU:** each model's L4 container stops 10 minutes after its last request.
  - Status polls read Modal's autoscaler stats (`Function.get_current_stats()`), which never reach the GPU container, so an open tab doesn't keep it awake.
- **Page:** it stops polling while hidden and after 5 minutes without a tap or key press. It then shows "Paused… tap anywhere to continue".
- **Frontend:** its CPU container stops 5 minutes later, so nothing runs 10 minutes after the last tap.
- **Broken model:** if a model fails to start, the frontend stops retrying and lets it scale down.
  - It stops after a minute once no page is waiting on that model, for example after a switch to the other model.
  - It stops after 20 minutes at most.
- **Billing:** Modal bills only while containers run. L4 time costs $0.000222/s (about $0.80/h), and the hybrid's 4 reserved CPU cores add a little.
  - The cached weights (about 9.4 GB for Voxtral and 0.46 GB for the hybrid, in Volumes) fall within the 1 TiB/month of free Volume storage.

## Deploy

```bash
# once: secrets (values never printed)
modal secret create huggingface-voxtral --from-json hf.json        # {"HF_TOKEN": "..."}, read access to the private model
modal secret create voxtral-bn-vllm-key --from-json key.json       # {"VLLM_API_KEY": "<random>"}, shared by all three apps

# once: cache the weights in Volumes (Voxtral's Mistral-format files; the hybrid's .nemo)
modal run modal_deployment/inference_app.py::download_weights
modal run modal_deployment/hybrid_inference_app.py::download_weights

modal deploy modal_deployment/inference_app.py
modal deploy modal_deployment/hybrid_inference_app.py
modal deploy modal_deployment/frontend_app.py   # prints the page URL: https://<workspace>--voxtral-bn-asr.<region>.modal.run
```

Redeploying the frontend with `--strategy recreate` makes the new version serve immediately.

## Use it from code

The frontend's API needs no key:

```bash
curl -F file=@speech.m4a https://<workspace>--voxtral-bn-asr.<region>.modal.run/api/transcribe                   # Voxtral Mini
curl -F file=@speech.m4a -F model=hybrid https://<workspace>--voxtral-bn-asr.<region>.modal.run/api/transcribe   # FastConformer hybrid
# {"text": "...", "pieces": [...], "model": "hybrid", "audio_seconds": 16.2, ...}   (503 while that model is not ready)
```

Both inference services need the key. The Voxtral service speaks vLLM's OpenAI-compatible API. Use `../vllm_inference/client.py --url <inference URL> --api-key-file <file>`, which also does the benchmark's chunking.

Accepted input: WAV, FLAC, MP3, OGG/Opus, M4A/AAC, WebM, and video files with an audio track, up to 10 minutes and 100 MB. Language is fixed to Bangla.
- **Size limits:** oversized bodies are refused while they arrive. The duration and decoded size are checked from the file header before decoding, so a tiny compressed file can't expand into gigabytes.
- **Playlists:** ffmpeg only opens audio and video containers, so playlists and concat lists are refused.

## Test

`python modal_deployment/test_frontend.py` drives the real web app against fake autoscalers and fake inference servers for both models. It needs fastapi, httpx, python-multipart, numpy, soundfile and librosa, but no Modal and no GPU. It runs 46 checks:
- every readiness transition: asleep, waking, warm-up, ready, a backend that vanishes right after a success, a restarted frontend, stats lag, scale-down;
- that only an explicit wake of Voxtral sends a warm-up, and that a wake loop nobody waits on stops;
- that the two models are woken, tracked and sent audio separately, and that unknown models are refused;
- that no audio reaches vLLM early and that bad uploads never wake the GPU;
- upload limits, decompression bombs, playlists, WebM and M4A decoding;
- cancellation of sibling requests when one piece fails.

`python modal_deployment/test_hybrid_server.py` drives the hybrid server with a stand-in for the NeMo model. It runs 20 checks:
- the warm-up before routing;
- auth and the model name, and that unauthenticated or oversized bodies are refused before they are read;
- that only 16 kHz mono PCM16 WAV pieces of at most 30 s reach the model, and reach it byte for byte;
- that concurrent pieces are decoded one at a time.
