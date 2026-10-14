# bn-BD ASR serving

Serving for the author's private Bangla (Bangladesh) speech-to-text fine-tunes, behind one web page. Each model runs as its own inference microservice, and a small CPU frontend chunks the audio and calls it. Today it serves one model:

| Model | Served with | What it is |
|---|---|---|
| [`ehzawad/Voxtral-Mini-3B-2507-bn-BD`](https://huggingface.co/ehzawad/Voxtral-Mini-3B-2507-bn-BD) | [vLLM](https://docs.vllm.ai) 0.31 | Voxtral Mini 3B, an audio LLM |

**Live demo: <https://ehzawad-56775--voxtral-bn-asr.us-east.modal.run>.** Record or upload Bangla speech and read the transcript. It works on phones too.
- **Cold start:** the GPU sleeps when idle, so the first visit wakes it, which takes about 1–3 minutes. The page stays locked until the model is ready.
- **Speed:** through the page, a 16 s clip takes about 2 s on the L4.
- **Idle cost:** nothing runs, and nothing is billed, 10 minutes after the last use.

## Layout

```
services/
  frontend/       the page and its API: model picker, readiness gating, chunking (Modal, CPU)
  voxtral-mini/   Voxtral Mini inference (Modal, vLLM on 1× L4), plus the same server and a batch tool for any CUDA GPU
shared/
  asr_core.py     the benchmark's audio loading and chunking, used by every service that chunks
```

| Service | Modal app | What it does |
|---|---|---|
| [`services/frontend/`](services/frontend/) | `voxtral-mini-bn-frontend`, CPU | Serves the page. Decodes and chunks uploads, wakes only the selected model, and keeps the page locked until that model is ready. |
| [`services/voxtral-mini/`](services/voxtral-mini/) | `voxtral-mini-bn-inference`, 1× L4 | Runs `vllm serve` with an OpenAI-compatible transcription route. The folder also has `serve.sh`, `client.py` and `transcribe.py` for any CUDA GPU. |

The services deploy and scale independently, talk over HTTPS with a shared API key, and scale to zero when idle. The Modal app names keep their first deployment's names, so the URLs stay the same.

## Adding a model

Every inference service answers the same contract, so the frontend treats them all alike:
- `GET /health` returns 200 only once the model is ready.
- `POST /v1/audio/transcriptions` takes one piece of at most 25 s as a 16 kHz PCM16 WAV, with OpenAI's form fields, and answers `{"text": ...}`. It requires `Authorization: Bearer $VLLM_API_KEY`.
- Its Modal web function is named `serve`.

To add a model:
1. Create `services/<model>/modal_app.py` and deploy it.
2. Add the model to `MODELS` in `services/frontend/frontend_core.py` and to `MODELS` in `services/frontend/static/index.html`.
3. Redeploy the frontend. The model picker appears once there are two models.

A FastConformer hybrid RNN-T/CTC service (NeMo, on its own L4, behind the same contract) was served this way until 2026-10-15. Its code is in commit `f43471c`, under `modal_deployment/hybrid_*`.

## Same input as the benchmark

Every path feeds the model exactly what the benchmark fed it:
1. **Audio:** it goes through [bengali-asr-benchmark](https://github.com/ehzawad/bengali-asr-benchmark)'s `asr_core` (`shared/asr_core.py`, a verbatim, hash-checked copy).
   - It is resampled to 16 kHz mono.
   - It is cut into pieces of at most 25 s at pauses (no VAD), staged as PCM16.
   - The pieces' texts are joined with single spaces.
2. **Voxtral Mini:** each piece becomes Voxtral's transcription prompt with language `bn`. Decoding is greedy, with at most 500 new tokens.

The measured accuracy below therefore carries over to these paths.

## Accuracy: Voxtral Mini (benchmark run)

Snapshot `d1c7aeb0cd18`, decoded at batch 1 on one A100 by the benchmark's own harness and scored with its scorer on SHA-256-verified audio. WER and CER are pooled. The comparator in the last column is the public model with the lowest observed WER on each set. Its paired difference has a 95% bootstrap interval over reference clusters, which is pointwise and not adjusted for multiple comparisons.

| Set | Clips | WER | CER | Comparator (WER) | Mini minus comparator, WER points |
|---|---:|---:|---:|---|---|
| FLEURS test | 920 | 14.73% | 4.78% | Conformer Large (hishab) 14.37% | +0.35 [-0.36, +1.06] inconclusive |
| Vaani test | 1,000 | 22.08% | 6.47% | ehzawad FastConformer-CTC 24.67% | -2.59 [-3.24, -1.95] lower |
| SPRING-INX R1 | 1,000 | 36.87% | 21.79% | ehzawad FastConformer-CTC 39.23% | -2.35 [-2.96, -1.76] lower |
| SPRING-INX R2 | 1,000 | 40.01% | 23.61% | ehzawad FastConformer-CTC 38.42% | +1.59 [-1.53, +5.71] inconclusive |
| Kathbath known | 1,000 | 10.77% | 2.25% | Conformer Large (hishab) 9.33% | +1.44 [+0.77, +2.12] higher |
| Kathbath unknown | 1,000 | 10.82% | 2.09% | Conformer Large (hishab) 8.79% | +2.03 [+1.37, +2.71] higher |
| IndicVoices conv (validation) | 1,000 | 13.40% | 4.66% | ehzawad FastConformer-CTC 12.61% | +0.79 [-0.04, +1.56] inconclusive |
| SUBAK.KO test (Bangladeshi) | 1,000 | 11.66% | 5.52% | Conformer Large (hishab) 13.64% | -1.99 [-2.74, -1.25] lower |
| BanSpeech (Bangladeshi) | 1,000 | 20.72% | 7.69% | Whisper Medium 23.59% | -2.87 [-3.82, -1.94] lower |
| Common Voice 17 test | 1,000 | 10.32% | 2.67% | Conformer Large (hishab) 7.35% | +2.97 [+2.24, +3.70] higher |
| Ben-10 validation (Bangladeshi dialects) | 1,000 | 69.97% | 38.41% | Conformer Large (hishab) 63.54% | +6.44 [+2.18, +11.40] higher |

Conformer Large's card lists Kathbath, SUBAK.KO and Common Voice as training data, so those comparisons are not clean. On SUBAK.KO and BanSpeech, Voxtral Mini also has the lowest error rate of the author's other private bn-BD fine-tunes (Canary 1B v2, Qwen3-ASR 1.7B, FastConformer hybrid, Whisper turbo), though not separably from Canary and Qwen3-ASR. All nine models' results, per-set latency and the method are in [bengali-asr-benchmark](https://github.com/ehzawad/bengali-asr-benchmark).

**Repetition loops.** On SPRING-INX R2, three recordings ran to the 500-token cap while repeating a word. They contribute 663 of that set's 8,143 word errors and stay in its score. On Ben-10's dialect speech, 18 recordings did the same (61.2% WER without them). `transcribe.py` flags such pieces (`hit_token_cap`); the HTTP paths don't report it yet.

**Seven of the eleven sets are Indian-source.** SUBAK.KO, BanSpeech and Ben-10 are Bangladeshi; Common Voice records no speaker region. The model card documents no training data, so overlap with any of these sets is not ruled out.

## Same outputs on every path: Voxtral Mini (FLEURS test, 920 clips)

Each path was checked against the benchmark's earlier batched decode of the same snapshot ([bengali-asr-benchmark@d64a69f](https://github.com/ehzawad/bengali-asr-benchmark/tree/d64a69f)).

| Path | Identical to the batched decode | WER |
|---|---:|---:|
| Benchmark, batched decode (A100, vLLM offline, eager) | — | 14.73% |
| Benchmark, batch-1 run behind the accuracy table (A100, vLLM offline, CUDA graphs) | 860 | 14.73% |
| Modal deployment (L4, through the public page's API) | 853 | 14.69% |
| `services/voxtral-mini/transcribe.py` (A100, CUDA graphs) | 875 | 14.72% |
| `serve.sh` + `client.py` (A100, first 300 clips) | 283 of 300 | — |

Different GPUs and engine settings change a few transcripts in both directions. Aggregate WER doesn't move.
