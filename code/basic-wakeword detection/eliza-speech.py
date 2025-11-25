import argparse
import inspect
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional
import subprocess
import wave
from piper import PiperVoice
import eliza

import numpy as np
import pyaudio
import requests
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
    default="models/hey_mycroft_v0.1.onnx",
    required=False,
)
parser.add_argument(
    "--inference_framework",
    help="Inference backend to use (try 'onnx' on Raspberry Pi; 'tflite' if you have tflite_runtime)",
    type=str,
    default="onnx",
    required=False,
)
parser.add_argument(
    "--vad_threshold",
    help="Voice activity detection threshold (0 disables VAD-driven capture)",
    type=float,
    default=0.5,
    required=False,
)

parser.add_argument(
    "--output_dir",
    help="Directory to store captured wav files",
    type=str,
    default="recordings",
    required=False,
)
parser.add_argument(
    "--transcribe_url",
    help="Whisper.cpp inference endpoint (e.g. http://127.0.0.1:8080/inference). Leave empty to skip transcription.",
    type=str,
    default="http://127.0.0.1:8080/inference",
    required=False,
)
parser.add_argument(
    "--transcribe_timeout",
    help="Seconds to wait for Whisper.cpp transcription responses",
    type=float,
    default=30.0,
    required=False,
)
parser.add_argument(
    "--playback_device",
    help="ALSA device string for aplay (e.g. 'hw:2,0' or 'plughw:Headphones,0'). Leave empty to use the default device.",
    type=str,
    default="",
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
    path_param = first_supported_param(
        Model.__init__,
        (
            "wakeword_model_paths",
            "wakeword_models",
            "wakeword_model_path",
            "wakeword_models_paths",
        ),
    )
    if path_param:
        model_kwargs[path_param] = [args.model_path]
    else:
        print("Model path supplied but this Model() signature has no wakeword path parameter.")

param = first_supported_param(
    Model.__init__,
    ("inference_framework", "backend", "inference_backend")
)
if param:
    model_kwargs[param] = args.inference_framework

vad_param = first_supported_param(Model.__init__, ("vad_threshold",))
if vad_param:
    model_kwargs[vad_param] = max(0.0, float(args.vad_threshold))
# else: installed Model has no explicit backend or vad kwargs; it will choose defaults internally.

owwModel = Model(**model_kwargs)

# ---------- Debounce config ----------
DETECTION_THRESHOLD = 0.5        # what you already implicitly use
DEBOUNCE_SECONDS = 1             # ignore repeat triggers for ~0.7 s

# How many frames is that at your CHUNK size?
frames_per_second = TARGET_RATE / CHUNK  # e.g. 16000 / 1280 = 12.5
DEBOUNCE_FRAMES = max(1, int(DEBOUNCE_SECONDS * frames_per_second))

# Track cooldown (in frames) for the active model only
cooldown_remaining = 0

# Recording / capture configuration
MIN_RECORDING_DURATION = 1.5  # seconds
MAX_RECORDING_DURATION = 10.0  # seconds
OUTPUT_DIR = Path(args.output_dir).expanduser() if args.output_dir else None
capture_enabled = OUTPUT_DIR is not None
if capture_enabled:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
recording_active = False
record_buffer = bytearray()
record_label: Optional[str] = None
record_started_at: Optional[datetime] = None
detection_started_at: Optional[float] = None

TRANSCRIBE_URL = args.transcribe_url.strip()
TRANSCRIBE_TIMEOUT = max(0.1, float(args.transcribe_timeout))
transcription_enabled = bool(TRANSCRIBE_URL)

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


def sanitize_label(label: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label)
    cleaned = cleaned.strip("_")
    return cleaned or "wakeword"


def write_wav(samples: bytes, sample_rate: int, dest: Path):
    with wave.open(str(dest), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(FORMAT))
        wf.setframerate(sample_rate)
        wf.writeframes(samples)


def request_transcription(file_path: Path) -> Optional[str]:
    """Send the captured wav to whisper.cpp and return the transcription."""
    if not transcription_enabled:
        return None
    try:
        with file_path.open("rb") as fh:
            files = {"file": (file_path.name, fh, "audio/wav")}
            data = {"response_format": "json"}
            resp = requests.post(
                TRANSCRIBE_URL,
                files=files,
                data=data,
                timeout=TRANSCRIBE_TIMEOUT,
            )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[TRANSCRIBE warning] HTTP error from Whisper server: {e}")
        return None

    text_payload = None
    try:
        payload = resp.json()
        if isinstance(payload, dict):
            text_payload = payload.get("text") or payload.get("transcription")
            if not text_payload:
                segments = payload.get("segments")
                if isinstance(segments, list):
                    joined = " ".join(
                        str(seg.get("text", "")).strip()
                        for seg in segments
                        if isinstance(seg, dict)
                    ).strip()
                    if joined:
                        text_payload = joined
        else:
            text_payload = resp.text.strip()
    except ValueError:
        text_payload = resp.text.strip()

    if text_payload:
        return str(text_payload).strip()
    return None


def current_vad_score(model) -> float:
    """Return the max VAD score from recent frames (or default if unavailable)."""
    vad = getattr(model, "vad", None)
    if not vad or not hasattr(vad, "prediction_buffer"):
        return 0.0
    try:
        vad_frames = list(vad.prediction_buffer)[-20:]
        if not vad_frames:
            return 0.0
        return float(np.max(vad_frames))
    except Exception:
        return 0.0


if __name__ == "__main__":
    preferred_label = Path(args.model_path).stem if args.model_path else None
    print("#" * 60)
    print("Listening for a single wakeword...")
    if preferred_label:
        print(f"Preferred model: {preferred_label}")
    print("#" * 60)

    prediction_key = None
    model_label = preferred_label or "wakeword"

    eliza = eliza.Eliza()
    eliza.load('doctor.txt')

    print(eliza.initial())

    try:
        voice = PiperVoice.load("./en_US-lessac-medium.onnx")

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
            frame_bytes = frame.tobytes()

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
                pass

            triggered_this_frame = False

            # Wake word detected
            if score > DETECTION_THRESHOLD and cooldown_remaining == 0:
                cooldown_remaining = DEBOUNCE_FRAMES
                triggered_this_frame = True
                print(f"[TRIGGER] Wakeword '{model_label}' detected (score={score:.3f})")
                aplay_cmd = ["aplay"]
                if args.playback_device:
                    aplay_cmd += ["-D", args.playback_device]
                aplay_cmd.append("./start_listening.wav")
                subprocess.run(aplay_cmd)
                

            if triggered_this_frame and capture_enabled and not recording_active:
                record_buffer = bytearray(frame_bytes)
                record_label = model_label
                record_started_at = datetime.now()
                recording_active = True
                detection_started_at = time.perf_counter()
            elif capture_enabled and recording_active:
                record_buffer.extend(frame_bytes)

            if capture_enabled and recording_active:
                elapsed = (time.perf_counter() - detection_started_at) if detection_started_at else 0.0
                vad_required = args.vad_threshold > 0
                vad_score = current_vad_score(owwModel) if vad_required else 0.0
                vad_pass = vad_score > args.vad_threshold if vad_required else False
                should_continue = (
                    elapsed < MIN_RECORDING_DURATION
                    or (vad_required and elapsed < MAX_RECORDING_DURATION and vad_pass)
                )

                if not should_continue:
                    timestamp = (record_started_at or datetime.now()).strftime("%Y%m%d-%H%M%S")
                    safe_label = sanitize_label(record_label or model_label)
                    dest = OUTPUT_DIR / f"{timestamp}_{safe_label}.wav"
                    try:
                        write_wav(bytes(record_buffer), TARGET_RATE, dest)

                        print(f"[CAPTURE] Saved {elapsed:.2f}s of audio to {dest}")

                        transcription = request_transcription(dest)

                        if transcription:
                            print(f"[TRANSCRIBE] {transcription}")
                            said = transcription
                            response = eliza.respond(said)
                            if response:
                                with wave.open("test.wav", "wb") as wav_file:
                                    voice.synthesize_wav(response, wav_file)
                            aplay_cmd = ["aplay"]
                            if args.playback_device:
                                aplay_cmd += ["-D", args.playback_device]
                            aplay_cmd.append("./test.wav")
                            subprocess.run(aplay_cmd)
                            print(f"[ELIZA] {response}")

                    except Exception as e:
                        print(f"[CAPTURE warning] Failed to process {dest}: {e}")
                    finally:
                        recording_active = False
                        record_buffer = bytearray()
                        record_label = None
                        record_started_at = None
                        detection_started_at = None

    except KeyboardInterrupt:
        print("\n" + eliza.final())
        print("\nExiting...")
    finally:
        try:
            mic_stream.stop_stream()
            mic_stream.close()
        except Exception:
            pass
        pa.terminate()
