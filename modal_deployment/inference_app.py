"""Inference microservice: Voxtral Mini bn-BD behind vLLM's OpenAI-compatible
server on one Modal L4 that scales to zero.

    modal run modal_deployment/inference_app.py::download_weights   # once: weights -> Volume
    modal deploy modal_deployment/inference_app.py

It serves POST /v1/audio/transcriptions (guarded by VLLM_API_KEY) and GET /health
(open; 200 only once the engine has loaded). The frontend app chunks audio the way
the benchmark did and sends each <= 25 s piece here as a 16 kHz PCM16 WAV, so every
request becomes exactly the prompt the benchmark decoded: vLLM reads that WAV with
soundfile (no resampling) and Voxtral's get_generation_prompt builds the same
mistral_common transcription request as voxtral_vllm_bench.py.
"""
import subprocess

import modal

APP_NAME = "voxtral-mini-bn-inference"
MODEL_REPO = "ehzawad/Voxtral-Mini-3B-2507-bn-BD"
MODEL_REVISION = "d1c7aeb0cd18f8a7a057bf756df5ee2ccb7aebcd"  # the benchmarked snapshot
SERVED_MODEL_NAME = "voxtral-mini-bn-bd"
VLLM_PORT = 8000
MINUTES = 60

# Eager mode skips torch.compile and CUDA-graph capture, for a shorter cold start.
# False gives faster decoding after a slower boot (compile artifacts are cached in
# the vllm-cache Volume). Both modes give the same FLEURS WER in the benchmark.
FAST_BOOT = True

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm[audio]==0.31.0",  # torch 2.13.0 (CUDA 13.0), as in the benchmark venv
        "mistral-common[audio]==1.12.0",
        "librosa==1.0.0",
        "soundfile==0.14.0",
        "huggingface_hub==1.33.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache = modal.Volume.from_name("voxtral-bn-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("voxtral-bn-vllm-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-voxtral")  # HF_TOKEN (private repo)
api_key = modal.Secret.from_name("voxtral-bn-vllm-key")  # VLLM_API_KEY, shared with the frontend

app = modal.App(APP_NAME)


@app.function(
    image=image,
    secrets=[hf_secret],
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=60 * MINUTES,
)
def download_weights():
    """Fetch only the Mistral-format files vLLM loads (the repo also ships HF shards)."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        MODEL_REPO,
        revision=MODEL_REVISION,
        allow_patterns=["consolidated.safetensors", "params.json", "tekken.json",
                        "README.md", "LICENSE*", "NOTICE"],
    )
    hf_cache.commit()
    print(f"cached {MODEL_REPO}@{MODEL_REVISION} at {path}")


@app.function(
    image=image,
    gpu="L4",
    secrets=[hf_secret, api_key],
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    scaledown_window=10 * MINUTES,  # idle time before the GPU is released
    timeout=10 * MINUTES,
    startup_timeout=15 * MINUTES,  # otherwise `timeout` also bounds container startup
    max_containers=1,  # caps spend: at most one L4 at a time
)
@modal.concurrent(max_inputs=32)  # concurrent pieces are batched by vLLM
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * MINUTES)
def serve():
    cmd = [
        "vllm", "serve", MODEL_REPO,
        "--revision", MODEL_REVISION,
        "--tokenizer-revision", MODEL_REVISION,
        "--served-model-name", SERVED_MODEL_NAME,
        "--tokenizer-mode", "mistral",
        "--config-format", "mistral",
        "--load-format", "mistral",
        "--max-model-len", "2048",
        "--max-num-seqs", "32",
        "--max-num-batched-tokens", "16384",
        "--gpu-memory-utilization", "0.90",
        "--limit-mm-per-prompt", '{"audio": 1}',
        "--generation-config", "vllm",  # requests carry their own sampling settings
        "--seed", "0",
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--uvicorn-log-level", "warning",
        "--enforce-eager" if FAST_BOOT else "--no-enforce-eager",
    ]
    subprocess.Popen(cmd)  # VLLM_API_KEY from the secret guards /v1/*
