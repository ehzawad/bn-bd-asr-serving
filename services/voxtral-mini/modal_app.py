"""Inference microservice: Voxtral Mini bn-BD (FP8 decoder) behind vLLM's
OpenAI-compatible server on one Modal L4 that scales to zero.

    modal run services/voxtral-mini/modal_app.py::download_weights   # once: weights -> Volume
    modal deploy services/voxtral-mini/modal_app.py

It serves POST /v1/audio/transcriptions (guarded by VLLM_API_KEY) and GET /health
(open; 200 only once the engine has loaded). The frontend app chunks audio the way
the benchmark did and sends each <= 25 s piece here as a 16 kHz PCM16 WAV, so every
request becomes exactly the prompt the benchmark decoded: vLLM reads that WAV with
soundfile (no resampling) and Voxtral's get_generation_prompt builds the same
mistral_common transcription request as voxtral_vllm_bench.py.

Cold start comes from a GPU memory snapshot (as in bn-voice-chat and ec-natural-chatbot).
A full boot (weights, torch.compile, CUDA graphs, warm-up) takes ~4.5 minutes and is the
same every time, so the container boots once, vLLM goes to sleep level 1 (weights to CPU
RAM, KV cache released) and Modal snapshots CPU + GPU memory; later cold starts restore
that and wake vLLM in seconds. Each deploy drops the snapshots: the first 2-3 wakes after
one are full boots while Modal makes them.

vLLM listens on localhost only, behind nginx: the snapshot needs vLLM's dev-mode /sleep
and /wake_up routes, which VLLM_API_KEY does not guard, so nginx passes on /v1/* and
/health and nothing else.
"""
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import modal

APP_NAME = "voxtral-mini-bn-inference"
MODEL_REPO = "ehzawad/Voxtral-Mini-3B-2507-bn-BD"
# Branch fp8: the text decoder in FP8 (E4M3 weights, one scale per output channel; activations
# quantized per token at run time), the audio encoder BF16. 6.1 GB instead of 9.3 GB; the L4
# (Ada) runs it on native FP8 kernels. The model card lists its WER next to main (BF16) on the
# benchmark's eleven sets.
MODEL_REVISION = "287697acc3abeda13de40ecae073e07b67947b9c"  # branch fp8
SERVED_MODEL_NAME = "voxtral-mini-bn-bd"
PUBLIC_PORT, VLLM_PORT = 8000, 8001
MINUTES = 60

# True (eager) skips torch.compile and CUDA-graph capture, for a shorter cold start, but
# FP8 then decodes ~12% slower than BF16 on the L4: its per-token activation quantization
# only fuses into the matmuls when compiled. Compiled with CUDA graphs, FP8 is 1.56x faster
# than BF16 (16 s piece: 1.56 s vs 2.43 s). Compile artifacts are cached in the vllm-cache
# Volume, so only the first boot pays the full compile.
FAST_BOOT = False

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
    .apt_install("nginx-light")
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        # Only model files live on the Volume; HF's logs stay on container disk (a snapshot
        # taken while a Volume file is open cannot be restored once that file is gone).
        "HF_HOME": "/root/.hf-home",
        "HF_HUB_CACHE": "/root/.cache/huggingface/hub",
        "VLLM_SERVER_DEV_MODE": "1",  # /sleep and /wake_up, which the snapshot is built around
        "TORCHINDUCTOR_COMPILE_THREADS": "1",  # parallel inductor workers can break snapshots
    })
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


VLLM_CMD = [
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
    "--enable-sleep-mode",
    "--host", "127.0.0.1",
    "--port", str(VLLM_PORT),
    "--uvicorn-log-level", "warning",
    "--enforce-eager" if FAST_BOOT else "--no-enforce-eager",
]  # VLLM_API_KEY from the secret guards /v1/*

NGINX_CONF = f"""
worker_processes 1;
pid /tmp/nginx.pid;
error_log /dev/stderr warn;
events {{ worker_connections 1024; }}
http {{
  access_log off;
  client_max_body_size 20m;
  client_body_temp_path /tmp/nginx-body;
  proxy_temp_path /tmp/nginx-proxy;
  fastcgi_temp_path /tmp/nginx-fastcgi;
  uwsgi_temp_path /tmp/nginx-uwsgi;
  scgi_temp_path /tmp/nginx-scgi;
  upstream vllm {{ server 127.0.0.1:{VLLM_PORT}; keepalive 32; }}
  server {{
    listen {PUBLIC_PORT};
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_read_timeout 300s;
    location = /health {{ proxy_pass http://vllm; }}
    location /v1/ {{ proxy_pass http://vllm; }}
    location / {{ return 404; }}  # /sleep, /wake_up, /collective_rpc ... stay inside
  }}
}}
"""


def _vllm(path: str, body: bytes | None = None, headers: dict | None = None, timeout: float = 300) -> bytes:
    req = urllib.request.Request(f"http://127.0.0.1:{VLLM_PORT}{path}", data=body,
                                 headers=headers or {}, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _healthy() -> bool:
    try:
        _vllm("/health", timeout=3)
        return True
    except Exception:
        return False


def _warm_up_request() -> tuple[bytes, dict]:
    """A 2 s noise piece as a multipart transcription request (the frontend's warm-up shape)."""
    import io
    import uuid

    import numpy as np
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.random.default_rng(0).normal(0, 0.01, 32000).astype(np.float32), 16000,
             subtype="PCM_16", format="WAV")
    boundary = uuid.uuid4().hex
    parts = b""
    for k, v in (("model", SERVED_MODEL_NAME), ("language", "bn"), ("temperature", "0"),
                 ("max_completion_tokens", "32")):
        parts += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    parts += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="w.wav"\r\n'
              f"Content-Type: audio/wav\r\n\r\n").encode() + buf.getvalue() + f"\r\n--{boundary}--\r\n".encode()
    import os
    return parts, {"Content-Type": f"multipart/form-data; boundary={boundary}",
                   "Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"}


@app.cls(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,  # the sleeping engine keeps its weights (~5.7 GB) in CPU RAM in the snapshot
    secrets=[hf_secret, api_key],
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    scaledown_window=10 * MINUTES,  # idle time before the GPU is released
    timeout=10 * MINUTES,
    startup_timeout=20 * MINUTES,  # a full boot (no snapshot yet) is ~4.5 minutes
    max_containers=1,  # caps spend: at most one L4 at a time
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=32)  # concurrent pieces are batched by vLLM
class Voxtral:
    @modal.enter(snap=True)
    def boot(self):
        """Before the snapshot: vLLM loaded, compiled, warmed, then asleep."""
        t0 = time.monotonic()
        self.vllm = subprocess.Popen(VLLM_CMD)
        while not _healthy():
            if self.vllm.poll() is not None:
                raise RuntimeError(f"vLLM exited with code {self.vllm.returncode}")
            if time.monotonic() - t0 > 18 * MINUTES:
                raise TimeoutError("vLLM not healthy in time")
            time.sleep(1)
        body, headers = _warm_up_request()
        for _ in range(2):  # a fresh engine's first request is ~1 s slower; pay it here
            _vllm("/v1/audio/transcriptions", body, headers)
        hf_cache.commit()
        _vllm("/sleep?level=1", b"")
        print(f"[voxtral] booted and asleep in {time.monotonic() - t0:.0f}s; snapshotting", flush=True)

    @modal.enter(snap=False)
    def restore(self):
        """After a restore (or straight after boot): wake vLLM, then open the public port."""
        t0 = time.monotonic()
        _vllm("/wake_up", b"")
        while not _healthy():
            if self.vllm.poll() is not None:
                raise RuntimeError(f"vLLM exited with code {self.vllm.returncode}")
            time.sleep(0.5)
        conf = Path("/tmp/nginx.conf")
        conf.write_text(NGINX_CONF)
        self.nginx = subprocess.Popen(["nginx", "-c", str(conf), "-g", "daemon off;"])
        threading.Thread(target=self._watchdog, daemon=True).start()
        print(f"[voxtral] awake in {time.monotonic() - t0:.1f}s", flush=True)

    def _watchdog(self):
        """A dead vLLM closes the port, so Modal replaces the container."""
        while self.vllm.poll() is None:
            time.sleep(5)
        print(f"[voxtral] vLLM exited ({self.vllm.returncode}); closing the port", flush=True)
        self.nginx.terminate()

    # Same URL as before the class: the label keeps <workspace>--voxtral-mini-bn-inference-serve.
    @modal.web_server(port=PUBLIC_PORT, startup_timeout=20 * MINUTES, label="voxtral-mini-bn-inference-serve")
    def serve(self):
        """nginx is already listening (restore() started it)."""

    @modal.exit()
    def stop(self):
        for proc in (getattr(self, "nginx", None), self.vllm):
            if proc is not None:
                proc.terminate()
