# Frontend service

**Live:** <https://ehzawad-56775--voxtral-bn-asr.us-east.modal.run>. This is the page and its API, on a small CPU container (Modal app `voxtral-mini-bn-frontend`). It never loads a model: it calls each model's inference service under [`../`](../).

```
                                                        ┌─HTTPS + API key─▶  voxtral-mini-bn-inference (1× L4)
 browser ──HTTPS──▶  voxtral-mini-bn-frontend  (CPU)  ──┤                    services/voxtral-mini/
                     page, model picker, readiness,     └─ ... one inference service per model in MODELS
                     chunking
```

- `modal_app.py` is the Modal wrapper. `frontend_core.py` is the web app, free of Modal so it can be tested offline. `static/index.html` is the page.
- It decodes and chunks uploads with the benchmark's `asr_core` (`../../shared/asr_core.py`), sends each ≤ 25 s piece to the selected model's `/v1/audio/transcriptions` and joins the texts.
- The models it offers are `MODELS` in `frontend_core.py` and in `static/index.html`. Every endpoint takes `?model=`, and the transcribe form takes a `model` field. The default is `voxtral`.
- It tracks each model's readiness separately. The model picker shows once there are two or more models.
- It scales to zero 5 minutes after its last request.

## Readiness: nothing is accepted until the selected model is ready

1. **On load:** opening the page wakes the selected model's inference service, and only that one. Switching models wakes the newly selected one. The picker is locked while recording or transcribing.
2. **While waking:** Record and Upload stay locked behind a "waking the model" notice. A Voxtral cold start takes about half a minute (a GPU-memory snapshot restore).
3. **Unlocking:** the controls unlock only after the model's `/health` has answered 200. For a model with `warm_up` set, such as Voxtral, the page also waits for one short warm-up transcription, because a fresh vLLM spends about 1.1 s more on its first request.
4. **Before each use:** the page asks the model directly (`/api/status?model=…&probe=1`). For an upload, it asks before sending. For a recording, it asks while recording, because people start talking the moment they tap, and two round trips to the GPU's region used to cut off the first second or more. A recording starts the instant the microphone opens, and is discarded if the check fails.
5. **On the server:** `/api/transcribe` decodes and checks the upload first, so a bad file never touches the GPU. It then asks the model again right before sending any audio, and refuses with 503 (starting the wake) if the model does not answer.
6. **During a recording:** the page keeps the GPU warm while someone is recording, so a long recording never meets a sleeping model. If the permission prompt stays open while the page pauses or the model goes away, the recording does not start.

## Cost: nothing runs while nobody uses it

- **GPU:** status polls read Modal's autoscaler stats (`Function.get_current_stats()`), which never reach the GPU container, so an open tab doesn't keep it awake.
- **Page:** it stops polling while hidden and after 5 minutes without a tap or key press. It then shows "Paused… tap anywhere to continue".
- **Frontend:** its CPU container stops 5 minutes later. The GPU stops 10 minutes after its last request, so nothing runs 10 minutes after the last tap.
- **Broken model:** if a model fails to start, the frontend stops retrying and lets it scale down. It stops after a minute once no page is waiting on that model, and after 20 minutes at most.

## Deploy

```bash
# once: the API key shared with the inference services (value never printed)
modal secret create voxtral-bn-vllm-key --from-json key.json       # {"VLLM_API_KEY": "<random>"}

modal deploy services/frontend/modal_app.py   # prints the page URL: https://<workspace>--voxtral-bn-asr.<region>.modal.run
```

Deploy the inference services first (see their READMEs). Redeploying the frontend with `--strategy recreate` makes the new version serve immediately.

## Use it from code

The frontend's API needs no key:

```bash
curl -F file=@speech.m4a https://<workspace>--voxtral-bn-asr.<region>.modal.run/api/transcribe
# {"text": "...", "pieces": [...], "model": "voxtral", "audio_seconds": 16.2, ...}   (503 while the model is not ready)
```

Accepted input: WAV, FLAC, MP3, OGG/Opus, M4A/AAC, WebM, and video files with an audio track, up to 10 minutes and 100 MB. Language is fixed to Bangla.
- **Size limits:** oversized bodies are refused while they arrive. The duration and decoded size are checked from the file header before decoding, so a tiny compressed file can't expand into gigabytes.
- **Playlists:** ffmpeg only opens audio and video containers, so playlists and concat lists are refused.

## Test

`python services/frontend/test_frontend.py` drives the real web app against fake autoscalers and fake inference servers, for Voxtral and a test-only second model. It needs fastapi, httpx, python-multipart, numpy, soundfile and librosa, but no Modal and no GPU. It runs 46 checks:
- every readiness transition: asleep, waking, warm-up, ready, a backend that vanishes right after a success, a restarted frontend, stats lag, scale-down;
- that only an explicit wake of a `warm_up` model sends a warm-up, and that a wake loop nobody waits on stops;
- that models are woken, tracked and sent audio separately, and that unknown models are refused;
- that no audio reaches a model early and that bad uploads never wake the GPU;
- upload limits, decompression bombs, playlists, WebM and M4A decoding;
- cancellation of sibling requests when one piece fails.
