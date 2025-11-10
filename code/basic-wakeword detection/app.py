import argparse
import inspect
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudio
from openwakeword.model import Model

# ---------- Args ----------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--chunk_size",
    help="How much audio (in number of samples) to predict on at once (at the device rate)",
    type=int,
    default=1280,
    required=False,
)
parser.add_argument(
    "--model_path",
    help="Path of a specific model to load",
    type=str,
    default="",
    required=False,
)
parser.add_argument(
    "--inference_framework",
    help="Inference backend to use (try 'onnx' on Raspberry Pi; 'tflite' if you have tflite_runtime)",
    type=str,
    default="onnx",
    required=False,
)
args = parser.parse_args()

# ---------- Optional: download model assets across oww versions ----------
try:
    from openwakeword import utils as oww_utils
    if hasattr(oww_utils, "download_models"):
        oww_utils.download_models()
    elif hasattr(oww_utils, "download_assets"):
        oww_utils.download_assets()
    # else: newer versions may auto-download on first use
except Exception as e:
    print("Skipping explicit model download:", e)

# ---------- Audio helpers ----------
FORMAT = pyaudio.paInt16
CHANNELS = 1
TARGET_RATE = 16000
CHUNK = args.chunk_size  # in frames at the stream rate

pa = pyaudio.PyAudio()


def list_input_devices(p):
    """Return list of (index, info) for input-capable devices."""
    out = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if int(info.get("maxInputChannels", 0)) > 0:
            out.append((i, info))
    return out


def pick_input_device(p):
    """Prefer a USB mic if present; else first input device."""
    candidates = list_input_devices(p)
    if not candidates:
        raise RuntimeError("No input (capture) devices found. Plug in a USB mic and check `arecord -l`.")
    for i, info in candidates:
        name = (info.get("name") or "").lower()
        if "usb" in name:
            return i, int(info.get("defaultSampleRate") or TARGET_RATE)
    i, info = candidates[0]
    return i, int(info.get("defaultSampleRate") or TARGET_RATE)

def to_16k(x: np.ndarray, src_rate: int) -> np.ndarray:
    """Cheap linear resampler to 16 kHz for wakeword use."""
    if src_rate == TARGET_RATE:
        return x
    factor = TARGET_RATE / float(src_rate)
    idxs = np.linspace(0, len(x) - 1, int(len(x) * factor), endpoint=True)
    return np.interp(idxs, np.arange(len(x)), x).astype(np.int16)

# ---------- Open mic robustly ----------
idx, device_rate = pick_input_device(pa)

# Try opening at 16k first (ideal), else fall back to device default rate
stream_rate = TARGET_RATE
try:
    mic_stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=stream_rate,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=idx,
    )
except OSError:
    stream_rate = device_rate
    mic_stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=stream_rate,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=idx,
    )

# ---------- Build Model kwargs across oww versions ----------
def first_supported_param(func, candidates):
    try:
        params = inspect.signature(func).parameters
        for c in candidates:
            if c in params:
                return c
    except Exception:
        pass
    return None

model_kwargs = {}
if args.model_path:
    model_kwargs["wakeword_models"] = [args.model_path]

param = first_supported_param(
    Model.__init__,
    ("inference_framework", "backend", "inference_backend")
)
if param:
    model_kwargs[param] = args.inference_framework
# else: installed Model has no explicit backend kwarg; it will choose its default internally.

owwModel = Model(**model_kwargs)

# ---------- Debounce config ----------
DETECTION_THRESHOLD = 0.5        # what you already implicitly use
DEBOUNCE_SECONDS = 0.7           # ignore repeat triggers for ~0.7 s

# How many frames is that at your CHUNK size?
frames_per_second = TARGET_RATE / CHUNK  # e.g. 16000 / 1280 = 12.5
DEBOUNCE_FRAMES = max(1, int(DEBOUNCE_SECONDS * frames_per_second))

# Track cooldown (in frames) for the active model only
cooldown_remaining = 0

# ---------- Loop ----------
def resolve_prediction_key(model, preferred_name: Optional[str]) -> Optional[str]:
    """Pick a single prediction buffer key, optionally matching the preferred name."""
    buffer_keys = []
    if hasattr(model, "prediction_buffer"):
        buffer_keys = list(getattr(model, "prediction_buffer", {}).keys())
    if not buffer_keys:
        return None
    if preferred_name:
        preferred_lower = preferred_name.lower()
        for key in buffer_keys:
            if preferred_lower in key.lower():
                return key
    return buffer_keys[0]


def format_model_label(source: Optional[str], fallback: str) -> str:
    if not source:
        return fallback
    try:
        return Path(source).stem or fallback
    except (TypeError, ValueError):
        return str(source)


def extract_score(model, prediction, key: Optional[str]):
    """Return the latest score for the selected model."""
    if key and hasattr(model, "prediction_buffer"):
        buf = getattr(model, "prediction_buffer", {}).get(key)
        if buf:
            try:
                return float(list(buf)[-1])
            except (TypeError, ValueError):
                pass
    if isinstance(prediction, dict):
        if key and key in prediction:
            return float(prediction[key])
        if len(prediction) == 1:
            return float(next(iter(prediction.values())))
    try:
        return float(prediction)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    preferred_label = Path(args.model_path).stem if args.model_path else None
    print("#" * 60)
    print("Listening for a single wakeword...")
    if preferred_label:
        print(f"Preferred model: {preferred_label}")
    print("#" * 60)

    prediction_key = None
    model_label = preferred_label or "wakeword"

    try:
        while True:
            # Read raw mic frames; be tolerant of occasional I/O hiccups
            try:
                raw = mic_stream.read(CHUNK, exception_on_overflow=False)
            except OSError as e:
                print(f"[Audio warning] read() failed: {e}. Retrying...")
                continue

            frame = np.frombuffer(raw, dtype=np.int16)
            if stream_rate != TARGET_RATE:
                frame = to_16k(frame, stream_rate)

            # Inference with safety net
            try:
                prediction = owwModel.predict(frame)
            except Exception as e:
                print(f"[OWW warning] predict() failed: {e}. Continuing...")
                continue

            if prediction_key is None:
                prediction_key = resolve_prediction_key(owwModel, preferred_label)
                model_label = format_model_label(prediction_key, model_label)

            score = extract_score(owwModel, prediction, prediction_key)
            if score is None:
                continue

            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                continue

            if score > DETECTION_THRESHOLD:
                cooldown_remaining = DEBOUNCE_FRAMES
                print(f"[TRIGGER] Wakeword '{model_label}' detected (score={score:.3f})")

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        try:
            mic_stream.stop_stream()
            mic_stream.close()
        except Exception:
            pass
        pa.terminate()
