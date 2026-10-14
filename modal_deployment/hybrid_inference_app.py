"""Inference microservice: FastConformer hybrid RNN-T/CTC bn-BD in NeMo on one Modal L4 that
scales to zero, next to the Voxtral Mini service and behind the same page.

    modal run modal_deployment/hybrid_inference_app.py::download_weights   # once: weights -> Volume
    modal deploy modal_deployment/hybrid_inference_app.py

It serves POST /v1/audio/transcriptions (guarded by the same VLLM_API_KEY as the Voxtral service)
and GET /health; the server itself is hybrid_core.py. The model decodes the way the benchmark ran
it: the .nemo snapshot it measured, torch and nemo_toolkit at the benchmark's pins, the model's
default RNN-T decoder (greedy), batch 1, pieces staged by the frontend exactly as asr_core.decode
stages them.
"""
from pathlib import Path

import modal

APP_NAME = "fastconformer-hybrid-bn-inference"
MODEL_REPO = "ehzawad/fastconformer-hybrid-rnnt-ctc-large-bn-BD"
MODEL_REVISION = "47a7c3d5e45c501d909d85c9d64c32d2f44ec037"  # the benchmarked snapshot
NEMO_FILE = "fastconformer-hybrid-rnnt-ctc-large-bn-BD.nemo"
MINUTES = 60

HERE = Path(__file__).parent
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "libsndfile1")
    # the benchmark's pins (bengali-asr-benchmark/requirements.txt): torch 2.11.0 (CUDA 12.8), NeMo 2.7.3
    .uv_pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install(
        "torch==2.11.0",  # already installed; keeps the resolver from replacing the CUDA 12.8 build
        "nemo_toolkit[asr]==2.7.3",
        "soundfile==0.14.0",
        "librosa==1.0.0",
        "fastapi[standard]==0.136.3",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_file(HERE / "hybrid_core.py", "/root/hybrid_core.py")
)

hf_cache = modal.Volume.from_name("fastconformer-hybrid-bn-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-voxtral")  # HF_TOKEN, read access to the private model
api_key = modal.Secret.from_name("voxtral-bn-vllm-key")  # VLLM_API_KEY, shared with the frontend

app = modal.App(APP_NAME)


@app.function(image=image, secrets=[hf_secret], volumes={"/root/.cache/huggingface": hf_cache},
              timeout=30 * MINUTES)
def download_weights():
    from huggingface_hub import hf_hub_download

    for name in (NEMO_FILE, "README.md", "LICENSE", "NOTICE"):
        path = hf_hub_download(MODEL_REPO, name, revision=MODEL_REVISION)
    hf_cache.commit()
    print(f"cached {MODEL_REPO}@{MODEL_REVISION} next to {path}")


class NeMoModel:
    """The benchmark's NeMo path: ASRModel.restore_from(.nemo) on CUDA in eval mode, and
    transcribe(paths, batch_size=1) under torch.inference_mode, as asr_core.decode calls it."""

    def __init__(self, path: str):
        import nemo.collections.asr as nemo_asr

        self.m = nemo_asr.models.ASRModel.restore_from(path).to("cuda").eval()
        print(f"loaded {type(self.m).__name__}, decoder {getattr(self.m, 'cur_decoder', '?')}", flush=True)

    def transcribe(self, paths, batch_size=1, verbose=False):
        import torch

        with torch.inference_mode():
            return self.m.transcribe(list(paths), batch_size=batch_size, verbose=verbose)


@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,  # reserved cores for NeMo's CPU-side work (manifest and data loading, decoding loop)
    secrets=[hf_secret, api_key],
    volumes={"/root/.cache/huggingface": hf_cache},
    scaledown_window=10 * MINUTES,  # idle time before the GPU is released, as for Voxtral
    timeout=10 * MINUTES,
    startup_timeout=10 * MINUTES,
    max_containers=1,  # caps spend: at most one L4 at a time
)
@modal.concurrent(max_inputs=32)  # pieces queue for the one model, decoded one at a time
@modal.asgi_app()
def serve():
    # Runs once per container, before any request is routed: requests (including /health)
    # wait here until the model has loaded and warmed up.
    import os

    from huggingface_hub import hf_hub_download

    import hybrid_core

    model = NeMoModel(hf_hub_download(MODEL_REPO, NEMO_FILE, revision=MODEL_REVISION))
    return hybrid_core.create_app(model, os.environ["VLLM_API_KEY"])
