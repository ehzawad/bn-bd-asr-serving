"""Audio handling shared by the web demo (app.py) and the benchmark endpoint
(bench_server.py).

It lives in its own module so that every model measured in the benchmark goes
through byte-identical preprocessing. If the demo and the benchmark each carried
their own copy, a divergence between them would show up as a model quality
difference, which is exactly the confound a benchmark exists to avoid.

Nothing here loads or holds a model: `decode` takes one as an argument, so a
caller can serve any .nemo checkpoint through the same path.
"""

import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

TARGET_SR = 16000
CHUNK_SECONDS = 25.0        # under the 30 s training cap
SEARCH_SECONDS = 2.0        # look this far *back* from a boundary for a pause
MIN_PIECE_SECONDS = 0.3     # train_ds min_duration is 0.1; keep clear of it


def resample(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz.

    librosa (soxr under the hood) is the backend, chosen explicitly rather than
    behind a try/except chain: a silent fallback would let a genuine failure
    change which resampler runs, and resampler choice is audible to the encoder.
    """
    if sr == TARGET_SR:
        return audio
    return librosa.resample(
        np.ascontiguousarray(audio, dtype=np.float32), orig_sr=sr, target_sr=TARGET_SR
    )


def downmix(audio: np.ndarray) -> tuple[np.ndarray, str | None]:
    """Mono-ise. Averaging is right for correlated stereo but annihilates
    opposite-polarity channels, so fall back to the loudest single channel when
    the average loses most of the energy."""
    if audio.shape[1] == 1:
        return audio[:, 0], None

    per_channel = np.sqrt((audio.astype(np.float64) ** 2).mean(axis=0))
    mixed = audio.mean(axis=1)
    mixed_rms = float(np.sqrt((mixed.astype(np.float64) ** 2).mean()))
    best = int(per_channel.argmax())

    if per_channel[best] > 0 and mixed_rms < 0.5 * float(per_channel[best]):
        return audio[:, best], (
            f"channels cancelled on downmix; used channel {best} of {audio.shape[1]}"
        )
    return mixed, None


def load_audio(source) -> tuple[np.ndarray, str | None]:
    """Read a path or file-like object as 16 kHz mono float32."""
    audio, sr = sf.read(source, dtype="float32", always_2d=True)
    audio, note = downmix(audio)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return resample(audio, sr), note


def _envelope(audio: np.ndarray, step: int) -> np.ndarray:
    usable = len(audio) // step * step
    if usable == 0:
        return np.empty(0, dtype=np.float32)
    return np.abs(audio[:usable]).reshape(-1, step).mean(axis=1)


def chunks(audio: np.ndarray) -> tuple[list[np.ndarray], int]:
    """Split into pieces of at most CHUNK_SECONDS, preferring a quiet frame.

    The search runs strictly backwards from each nominal boundary, so a piece is
    never longer than the cap. Returns the pieces and the number of cuts that had
    to be forced through speech.
    """
    span = int(CHUNK_SECONDS * TARGET_SR)
    if len(audio) <= span:
        return [audio], 0

    step = int(0.01 * TARGET_SR)  # 10 ms frames
    rms = float(np.sqrt((audio.astype(np.float64) ** 2).mean()))
    quiet = 0.15 * rms  # "plausibly a pause" relative to this clip's own level

    pieces: list[np.ndarray] = []
    forced = 0
    start = 0
    while start < len(audio):
        if len(audio) - start <= span:
            pieces.append(audio[start:])
            break

        end = start + span
        lo = max(start + step, end - int(SEARCH_SECONDS * TARGET_SR))
        env = _envelope(audio[lo:end], step)
        cut = end
        if len(env):
            idx = int(env.argmin())
            if env[idx] <= quiet:
                cut = lo + idx * step
            else:
                forced += 1
        else:
            forced += 1

        pieces.append(audio[start:cut])
        start = cut

    # A sub-minimum tail is outside the model's training domain; fold it back.
    min_len = int(MIN_PIECE_SECONDS * TARGET_SR)
    if len(pieces) > 1 and len(pieces[-1]) < min_len:
        pieces[-2] = np.concatenate([pieces[-2], pieces[-1]])
        pieces.pop()
    return [p for p in pieces if len(p) > 0], forced


def decode(model, pieces: list[np.ndarray]) -> str:
    """Stage the normalised pieces as WAVs and run the model over them."""
    with tempfile.TemporaryDirectory(prefix="stt_bn_") as tmpdir:
        paths = []
        for i, piece in enumerate(pieces):
            # PCM16 is what NeMo's loader is exercised with; clip rather than let
            # an out-of-range float container wrap around on conversion.
            safe = np.clip(piece, -1.0, 1.0)
            p = str(Path(tmpdir) / f"chunk_{i:03d}.wav")
            sf.write(p, safe, TARGET_SR, subtype="PCM_16")
            paths.append(p)

        with torch.inference_mode():
            results = model.transcribe(paths, batch_size=1, verbose=False)

    texts = [(r.text if hasattr(r, "text") else str(r)).strip() for r in results]
    return " ".join(t for t in texts if t)
