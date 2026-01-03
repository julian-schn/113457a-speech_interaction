import argparse
import inspect
import subprocess
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudio
import requests
from openwakeword.model import Model

from piper import PiperVoice
import eliza


# ---------- Args ----------
parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    type=str,
    default="live",
    choices=["live", "offline_neg"],
    help="Run mode: live mic or offline negative evaluation"
)

parser.add_argument(
    "--neg_dir",
    type=str,
    default="",
    help="Directory with negative WAV files (offline_neg mode)"
)

parser.add_argument(
    "--threshold",
    type=float,
    default=0.5,
    help="Detection threshold for wakeword score"
)

parser.add_argument(
    "--hop_ms",
    type=float,
    default=80.0,
    help="Hop size in ms for offline audio scanning"
)

parser.add_argument(
    "--chunk_size",
    help="How much audio (in number of samples) to predict on at once (at the device rate)",
    type=int,
    default=1280,
    required=False,
)
parser.add_argument(
    "--model_path",
    help="Path of a specific model to load (e.g., ./model/mycroft.onnx)",
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
parser.add_argument(
    "--capture_seconds",
    help="How long after a trigger to keep recording audio before saving (0 disables saving)",
    type=float,
    default=2.0,
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
    default="",
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
except Exception as e:
    print("Skipping explicit model download:", e)


# ---------- Audio helpers ----------
FORMAT = pyaudio.paInt16
CHANNELS = 1
TARGET_RATE = 16000
CHUNK = args.chunk_size  # in frames at the stream rate

# IMPORTANT: Mic / PyAudio are initialized ONLY in live mode
pa = None
mic_stream = None
stream_rate = TARGET_RATE
idx = None
device_rate = None


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
        raise RuntimeError("No input (capture) devices found. Plug in a USB mic and check device settings.")
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


def init_mic_if_live():
    """Initialize PyAudio + mic stream only for live mode."""
    global pa, mic_stream, stream_rate, idx, device_rate

    pa = pyaudio.PyAudio()

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


# ---------- Build Model (FIXED: always supports explicit model path) ----------
# Your previous signature-inspection failed on your installed version.
# This version uses the canonical kwarg: wakeword_models=[...]
model_kwargs = {}

# Inference backend (onnx/tflite)
# Some versions may ignore this; harmless if unsupported.
model_kwargs["inference_framework"] = args.inference_framework

# Load ONLY the specified model when provided
if args.model_path:
    model_kwargs["wakeword_models"] = [args.model_path]

owwModel = Model(**model_kwargs)

# Debug to confirm what is loaded
try:
    print("[DEBUG] model_path arg:", args.model_path or "(default models)")
    print("[DEBUG] prediction_buffer keys:",
          list(getattr(owwModel, "prediction_buffer", {}).keys()))
except Exception as e:
    print("[DEBUG] could not read prediction_buffer keys:", e)


# ---------- Debounce config ----------
DETECTION_THRESHOLD = float(args.threshold)
DEBOUNCE_SECONDS = 1

frames_per_second = TARGET_RATE / CHUNK
DEBOUNCE_FRAMES = max(1, int(DEBOUNCE_SECONDS * frames_per_second))

cooldown_remaining = 0

# Recording / capture configuration (live mode)
CAPTURE_SECONDS = max(0.0, float(args.capture_seconds))
CAPTURE_SAMPLES_TARGET = int(CAPTURE_SECONDS * TARGET_RATE)
OUTPUT_DIR = Path(args.output_dir).expanduser()
capture_enabled = CAPTURE_SAMPLES_TARGET > 0
if capture_enabled:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
recording_active = False
record_buffer = bytearray()
record_samples = 0
record_label: Optional[str] = None
record_started_at: Optional[datetime] = None

TRANSCRIBE_URL = args.transcribe_url.strip()
TRANSCRIBE_TIMEOUT = max(0.1, float(args.transcribe_timeout))
transcription_enabled = bool(TRANSCRIBE_URL)


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


# ---------- OFFLINE NEG ----------
def read_wav_int16_mono(path: Path):
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        raise ValueError(f"{path} is not 16-bit PCM")

    x = np.frombuffer(raw, dtype=np.int16)

    if nch == 2:
        x = x.reshape(-1, 2).mean(axis=1).astype(np.int16)

    if sr != TARGET_RATE:
        x = to_16k(x, sr)

    return x


def run_offline_negative_eval(neg_dir: Path):
    files = sorted(p for p in neg_dir.rglob("*.wav"))
    if not files:
        raise RuntimeError(f"No WAV files in {neg_dir}")

    hop_samples = max(1, int(TARGET_RATE * args.hop_ms / 1000))
    chunk_samples = CHUNK

    prediction_key = None

    # Optional warmup (helps first-call latency)
    try:
        _ = owwModel.predict(np.zeros(CHUNK, dtype=np.float32))
    except Exception:
        pass

    total_seconds = 0.0
    false_alarms = 0

    for idx_f, wav in enumerate(files, start=1):
        if idx_f == 1 or idx_f % 10 == 0:
            print(f"[OFFLINE_NEG] file {idx_f}/{len(files)}: {wav.name}")

        audio = read_wav_int16_mono(wav)
        total_seconds += len(audio) / TARGET_RATE

        cooldown_remaining_local = 0
        i = 0
        while i + chunk_samples <= len(audio):
            frame = audio[i:i + chunk_samples]

            # IMPORTANT: openwakeword expects float32 audio in [-1, 1]
            frame_f = frame.astype(np.float32) / 32768.0

            prediction = owwModel.predict(frame_f)

            if prediction_key is None:
                prediction_key = resolve_prediction_key(
                    owwModel,
                    Path(args.model_path).stem if args.model_path else None
                )
                if prediction_key is None:
                    raise RuntimeError("Could not resolve prediction key (no prediction_buffer keys).")
                print(f"[DEBUG] using key: {prediction_key}")

            score = extract_score(owwModel, prediction, prediction_key)

            if cooldown_remaining_local > 0:
                cooldown_remaining_local -= 1

            if score is not None and score >= DETECTION_THRESHOLD and cooldown_remaining_local == 0:
                false_alarms += 1
                cooldown_remaining_local = DEBOUNCE_FRAMES

            i += hop_samples

    hours = total_seconds / 3600
    far = false_alarms / hours if hours > 0 else float("nan")

    print("\n" + "#" * 60)
    print("OFFLINE NEGATIVE EVALUATION")
    print(f"Files           : {len(files)}")
    print(f"Audio duration  : {hours:.2f} h")
    print(f"False alarms    : {false_alarms}")
    print(f"FAR             : {far:.6f} FA/h")
    print(f"Threshold       : {DETECTION_THRESHOLD}")
    print(f"hop_ms          : {args.hop_ms}")
    print("#" * 60 + "\n")


if __name__ == "__main__":

    if args.mode == "offline_neg":
        if not args.neg_dir:
            raise RuntimeError("--neg_dir required for offline_neg mode")

        run_offline_negative_eval(Path(args.neg_dir))
        raise SystemExit(0)

    # Live mode: initialize mic only here
    init_mic_if_live()

    preferred_label = Path(args.model_path).stem if args.model_path else None
    print("#" * 60)
    print("Listening for a single wakeword...")
    if preferred_label:
        print(f"Preferred model: {preferred_label}")
    print("#" * 60)

    prediction_key = None
    model_label = preferred_label or "wakeword"

    try:
        voice = PiperVoice.load("./en_US-lessac-medium.onnx")

        el = eliza.Eliza()
        el.load('doctor.txt')

        while True:
            try:
                raw = mic_stream.read(CHUNK, exception_on_overflow=False)
            except OSError as e:
                print(f"[Audio warning] read() failed: {e}. Retrying...")
                continue

            frame = np.frombuffer(raw, dtype=np.int16)
            if stream_rate != TARGET_RATE:
                frame = to_16k(frame, stream_rate)
            frame_bytes = frame.tobytes()
            samples_in_frame = len(frame)

            try:
                # FIX: normalize like offline
                frame_f = frame.astype(np.float32) / 32768.0
                prediction = owwModel.predict(frame_f)
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

            triggered_this_frame = False

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
                record_samples = samples_in_frame
                record_label = model_label
                record_started_at = datetime.now()
                recording_active = True
            elif capture_enabled and recording_active:
                record_buffer.extend(frame_bytes)
                record_samples += samples_in_frame

            if capture_enabled and recording_active and record_samples >= CAPTURE_SAMPLES_TARGET:
                timestamp = (record_started_at or datetime.now()).strftime("%Y%m%d-%H%M%S")
                safe_label = sanitize_label(record_label or model_label)
                dest = OUTPUT_DIR / f"{timestamp}_{safe_label}.wav"
                try:
                    write_wav(bytes(record_buffer), TARGET_RATE, dest)
                    print(f"[CAPTURE] Saved {CAPTURE_SECONDS:.2f}s of audio to {dest}")
                    transcription = request_transcription(dest)
                    if transcription:
                        with wave.open("test.wav", "wb") as wav_file:
                            voice.synthesize_wav(transcription, wav_file)
                        aplay_cmd = ["aplay"]
                        if args.playback_device:
                            aplay_cmd += ["-D", args.playback_device]
                        aplay_cmd.append("./test.wav")
                        subprocess.run(aplay_cmd)
                    print(f"[TRANSCRIBE] {transcription}")
                except Exception as e:
                    print(f"[CAPTURE warning] Failed to process {dest}: {e}")
                finally:
                    recording_active = False
                    record_buffer = bytearray()
                    record_samples = 0
                    record_label = None
                    record_started_at = None

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        try:
            if mic_stream is not None:
                mic_stream.stop_stream()
                mic_stream.close()
        except Exception:
            pass
        try:
            if pa is not None:
                pa.terminate()
        except Exception:
            pass
