# Voxtral Mini bn-BD inference

Serving and fast batch inference for [`ehzawad/Voxtral-Mini-3B-2507-bn-BD`](https://huggingface.co/ehzawad/Voxtral-Mini-3B-2507-bn-BD) (private), a Bangla (Bangladesh) speech-to-text fine-tune of Voxtral Mini 3B, with [vLLM](https://docs.vllm.ai) 0.31.

**Live demo: <https://ehzawad-56775--voxtral-bn-asr.us-east.modal.run>.** Record or upload Bangla speech and get the transcript; it works on phones too.
- **Cold start:** the GPU sleeps when idle, so the first visit wakes it, which takes about 1–3 minutes. The page stays locked until the model is ready.
- **Idle cost:** nothing runs, and nothing is billed, 10 minutes after the last use.

| Folder | What it is |
|---|---|
| [`modal_deployment/`](modal_deployment/) | Two Modal microservices: a vLLM inference service on one NVIDIA L4 and a small web page where people record or upload Bangla speech. Both scale to zero after idle minutes. |
| [`vllm_inference/`](vllm_inference/) | Local tools for any CUDA GPU: `transcribe.py` (fast offline batch transcription), `serve.sh` (vLLM's OpenAI-compatible server with the deployment's settings) and `client.py` (chunks long audio and calls any such server, including the Modal one). |

All three paths feed the model exactly what the benchmark fed it:
1. Audio goes through [bengali-asr-benchmark](https://github.com/ehzawad/bengali-asr-benchmark)'s `asr_core` (a verbatim, hash-checked copy): 16 kHz mono, pieces of at most 25 s cut at pauses (no VAD), PCM16 staging.
2. Each piece becomes Voxtral's transcription prompt with language `bn`.
3. Decoding is greedy, with at most 500 new tokens.
4. The pieces' texts are joined with single spaces.

The measured accuracy below therefore carries over to these paths.

## Accuracy (benchmark run)

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

## Same outputs on every path (FLEURS test, 920 clips)

Each path was checked against the benchmark's earlier batched decode of the same snapshot ([bengali-asr-benchmark@d64a69f](https://github.com/ehzawad/bengali-asr-benchmark/tree/d64a69f)).

| Path | Identical to the batched decode | WER |
|---|---:|---:|
| Benchmark, batched decode (A100, vLLM offline, eager) | — | 14.73% |
| Benchmark, batch-1 run behind the accuracy table (A100, vLLM offline, CUDA graphs) | 860 | 14.73% |
| Modal deployment (L4, through the public page's API) | 853 | 14.69% |
| `vllm_inference/transcribe.py` (A100, CUDA graphs) | 875 | 14.72% |
| `serve.sh` + `client.py` (A100, first 300 clips) | 283 of 300 | — |

Different GPUs and engine settings change a few transcripts in both directions. Aggregate WER doesn't move.
