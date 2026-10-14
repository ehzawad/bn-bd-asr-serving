# bn-BD ASR serving

Serving for two of the author's private Bangla (Bangladesh) speech-to-text fine-tunes, behind one web page:

| Model | Served with | What it is |
|---|---|---|
| [`ehzawad/Voxtral-Mini-3B-2507-bn-BD`](https://huggingface.co/ehzawad/Voxtral-Mini-3B-2507-bn-BD) | [vLLM](https://docs.vllm.ai) 0.31 | Voxtral Mini 3B, an audio LLM |
| [`ehzawad/fastconformer-hybrid-rnnt-ctc-large-bn-BD`](https://huggingface.co/ehzawad/fastconformer-hybrid-rnnt-ctc-large-bn-BD) | [NeMo](https://github.com/NVIDIA/NeMo) 2.7.3, RNN-T decoding | FastConformer hybrid RNN-T/CTC, 115M parameters |

**Live demo: <https://ehzawad-56775--voxtral-bn-asr.us-east.modal.run>.** Pick a model, record or upload Bangla speech, and read the transcript. It works on phones too.
- **Two models:** each runs on its own NVIDIA L4, and the page wakes only the selected one. Open `?model=hybrid` to start on the hybrid. After a transcript, the other model can transcribe the same audio.
- **Cold start:** a GPU sleeps when idle, so the first visit wakes it. That takes about 1–3 minutes for Voxtral Mini and about a minute for the hybrid. The page stays locked until the model is ready.
- **Speed:** through the page, a 16 s clip takes under a second with the hybrid (0.3–0.9 s) and about 2 s with Voxtral Mini.
- **Idle cost:** nothing runs, and nothing is billed, 10 minutes after the last use.

| Folder | What it is |
|---|---|
| [`modal_deployment/`](modal_deployment/) | Three Modal microservices: a vLLM inference service for Voxtral Mini on one L4, a NeMo inference service for the hybrid on another, and a small web page where people pick a model and record or upload Bangla speech. All three scale to zero after idle minutes. |
| [`vllm_inference/`](vllm_inference/) | Local tools for Voxtral Mini on any CUDA GPU: `transcribe.py` (fast offline batch transcription), `serve.sh` (vLLM's OpenAI-compatible server with the deployment's settings) and `client.py` (chunks long audio and calls any such server, including the Modal one). |

Every path feeds each model exactly what the benchmark fed it:
1. **Audio:** it goes through [bengali-asr-benchmark](https://github.com/ehzawad/bengali-asr-benchmark)'s `asr_core` (a verbatim, hash-checked copy).
   - It is resampled to 16 kHz mono.
   - It is cut into pieces of at most 25 s at pauses (no VAD), staged as PCM16.
   - The pieces' texts are joined with single spaces.
2. **Voxtral Mini:** each piece becomes Voxtral's transcription prompt with language `bn`. Decoding is greedy, with at most 500 new tokens.
3. **FastConformer hybrid:** each piece's WAV goes unchanged to NeMo's `transcribe`.
   - It uses the model's default RNN-T decoder (greedy), at batch 1.
   - It runs the benchmarked snapshot, with the benchmark's torch and NeMo pins.

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

## Accuracy: FastConformer hybrid (benchmark run)

Snapshot `47a7c3d5`, decoded and scored by the same harness on the same audio. The rank is among the benchmark's nine models, by WER.

| Set | Clips | WER (95% CI) | CER | Rank of 9 | Voxtral Mini WER |
|---|---:|---:|---:|---:|---:|
| FLEURS test | 920 | 15.38% [14.26–16.54] | 4.91% | 6 | 14.73% |
| Vaani test | 1,000 | 22.53% [21.4–23.7] | 6.74% | 4 | 22.08% |
| SPRING-INX R1 | 1,000 | 36.92% [35.7–38.2] | 21.83% | 4 | 36.87% |
| SPRING-INX R2 | 1,000 | 36.60% [35.4–37.9] | 20.57% | 2 | 40.01% |
| Kathbath known | 1,000 | 11.60% [10.8–12.4] | 2.33% | 7 | 10.77% |
| Kathbath unknown | 1,000 | 11.41% [10.6–12.2] | 2.22% | 7 | 10.82% |
| IndicVoices conv (validation) | 1,000 | 12.70% [11.9–13.6] | 4.54% | 3 | 13.40% |
| SUBAK.KO test (Bangladeshi) | 1,000 | 14.87% [13.6–16.2] | 7.93% | 7 | 11.66% |
| BanSpeech (Bangladeshi) | 1,000 | 23.47% [22.0–25.0] | 9.27% | 5 | 20.72% |
| Common Voice 17 test | 1,000 | 10.51% [9.7–11.4] | 2.47% | 7 | 10.32% |
| Ben-10 validation (Bangladeshi dialects) | 1,000 | **60.91%** [59.7–62.1] | 33.05% | **1** | 69.97% |

The intervals are 95% bootstraps over reference clusters; the benchmark has the paired comparisons.
- **Where it leads:** the hybrid is the best of the nine on Ben-10's dialect speech and second on SPRING-INX R2.
- **Speed:** it ran about 4× faster than Voxtral Mini in the benchmark, 0.10 s against 0.44 s per clip on an A100.
- **Empty transcripts:** it returned 23 on SUBAK.KO, which stay in that set's score.

On the deployment, the hybrid service returns the benchmark's exact transcript for FLEURS test clip 0001, both sent directly and through the page. A full-set comparison like Voxtral Mini's below has not been run for it yet.

## Same outputs on every path: Voxtral Mini (FLEURS test, 920 clips)

Each path was checked against the benchmark's earlier batched decode of the same snapshot ([bengali-asr-benchmark@d64a69f](https://github.com/ehzawad/bengali-asr-benchmark/tree/d64a69f)).

| Path | Identical to the batched decode | WER |
|---|---:|---:|
| Benchmark, batched decode (A100, vLLM offline, eager) | — | 14.73% |
| Benchmark, batch-1 run behind the accuracy table (A100, vLLM offline, CUDA graphs) | 860 | 14.73% |
| Modal deployment (L4, through the public page's API) | 853 | 14.69% |
| `vllm_inference/transcribe.py` (A100, CUDA graphs) | 875 | 14.72% |
| `serve.sh` + `client.py` (A100, first 300 clips) | 283 of 300 | — |

Different GPUs and engine settings change a few transcripts in both directions. Aggregate WER doesn't move.
