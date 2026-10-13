"""Frontend microservice: one page where people record or upload Bangla (Bangladesh)
speech and read Voxtral Mini bn-BD's transcript. CPU only; it holds no model and calls
the inference app over HTTP. The web app itself lives in frontend_core.py.

    modal deploy modal_deployment/frontend_app.py

The page wakes the inference service when it loads and stays locked until vLLM reports
the model ready; it stops polling after 10 idle minutes or while hidden, so both apps can
scale to zero and nothing is billed while nobody uses it.
"""
from pathlib import Path

import modal

APP_NAME = "voxtral-mini-bn-frontend"
INFERENCE_APP = "voxtral-mini-bn-inference"
MINUTES = 60

HERE = Path(__file__).parent
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")  # browser recordings arrive as WebM/Opus or MP4/AAC
    .uv_pip_install(
        "fastapi[standard]==0.136.3",
        "httpx==0.28.1",
        "numpy==2.3.5",
        "soundfile==0.14.0",
        "librosa==1.0.0",
        "soxr==1.1.0",
    )
    .add_local_file(HERE / "asr_core.py", "/root/asr_core.py")
    .add_local_file(HERE / "frontend_core.py", "/root/frontend_core.py")
    .add_local_dir(HERE / "static", "/root/static")
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("voxtral-bn-vllm-key")],  # VLLM_API_KEY, shared with inference
    scaledown_window=5 * MINUTES,  # idle time before this CPU container stops
    timeout=15 * MINUTES,
    max_containers=1,  # one process holds the wake state
)
@modal.concurrent(max_inputs=64)
@modal.asgi_app(label="voxtral-bn-asr")
def web():
    import os

    import frontend_core

    return frontend_core.create_app(
        serve_fn=modal.Function.from_name(INFERENCE_APP, "serve"),
        api_key=os.environ["VLLM_API_KEY"],
        asr_core=frontend_core.load_asr_core(Path("/root/asr_core.py")),
        static_dir=Path("/root/static"),
    )
