"""Frontend microservice: one page where people record or upload Bangla (Bangladesh) speech,
pick a model, and read the transcript. CPU only; it holds no model and calls each model's
inference service over HTTP. The web app itself lives in frontend_core.py, and the models it
offers are its MODELS registry.

    modal deploy services/frontend/modal_app.py

The page wakes only the selected model's inference service when it loads (or when the model is
switched) and stays locked until that model reports ready; it stops polling after 5 idle
minutes or while hidden, so every service can scale to zero and nothing is billed while nobody
uses it.
"""
from pathlib import Path

import modal

APP_NAME = "voxtral-mini-bn-frontend"  # kept from the first deployment, with the page's URL label
VOICE_CHAT_APP = "bn-chat"  # the Bangla voice chatbot (repo bn-voice-chat); /chat here forwards to it
MINUTES = 60

HERE = Path(__file__).parent
# Only `modal deploy` reads it. The container imports this file as /root/modal_app.py, where
# parents[1] does not exist, so this must stay plain .parent arithmetic.
SHARED = HERE.parent.parent / "shared"
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
    .add_local_file(SHARED / "asr_core.py", "/root/asr_core.py")
    .add_local_file(HERE / "frontend_core.py", "/root/frontend_core.py")
    .add_local_dir(HERE / "static", "/root/static")
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("voxtral-bn-vllm-key")],  # VLLM_API_KEY, shared with the inference services
    scaledown_window=5 * MINUTES,  # idle time before this CPU container stops
    timeout=15 * MINUTES,
    max_containers=1,  # one process holds the wake state
)
@modal.concurrent(max_inputs=64)
@modal.asgi_app(label="voxtral-bn-asr")
def web():
    import os

    import frontend_core

    def serve_fn(m):
        # A snapshotted service is a class whose web endpoint is its `serve` method.
        if m.get("cls"):
            return modal.Cls.from_name(m["app"], m["cls"])().serve
        return modal.Function.from_name(m["app"], "serve")

    async def voice_chat_url():
        url = await modal.Function.from_name(VOICE_CHAT_APP, "web").get_web_url.aio()
        return url.rstrip("/") + "/chat"

    return frontend_core.create_app(
        serve_fns={name: serve_fn(m) for name, m in frontend_core.MODELS.items()},
        voice_chat_url=voice_chat_url,
        api_key=os.environ["VLLM_API_KEY"],
        asr_core=frontend_core.load_asr_core(Path("/root/asr_core.py")),
        static_dir=Path("/root/static"),
    )
