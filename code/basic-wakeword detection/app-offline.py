import argparse
import inspect
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pyaudio
import requests
from openwakeword.model import Model

from piper import PiperVoice
import eliza


# ============================================================
# Args
# ============================================================
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
    help="Directory with WAV files (offline_neg mode). You can also put pos_wavs here for a sanity check."
)

parser.add_argument(
    "--threshold",
    type=float,
    default=0.5,
    help="Detection threshold for wakeword score"
)

parser.add_argument(
    "--release_ratio",
    type=float,
    default=0.5,
    help="Peak-picking release ratio: next trigger is armed when score < threshold*release_ratio"
)

parser.add_argument(
    "--hop_ms",
    type=float,
    default=80.0,
    help="Hop size in ms for offline audio scanning"
)

parser.add_argument(
    "--chunk_size",
    help="How many samples to predict on at once",
    type=int,
    default=1280,
    required=False,
)

parser.add_argument(
    "--model_path",
    help="Path of a specific model to load (e.g., ./models/hey_mycroft.onnx)",
    type=str,
    default="",
    required=False,
)

parser.add_argument(
    "--inference_framework",
    help="Inference backend to use (onnx/tflite). Some versions may ignore this.",
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
    help="ALSA device string for aplay (Linux). Leave empty to use default device.",
    type=str,
    default="",
    required=False,
)

parser.add_argument(
    "--mic_device",
    help="PyAudio input device index (Windows: choose Microphone Array etc.). -1 = auto pick.",
    type=int,
    default=-1,
    required=False,
)

parser.add_argument(
    "--list_devices",
    help="Print input devices and exit",
    action="store_true",
)

args = parser.parse_args()


# ============================================================
# Optional: download model assets across openwakeword versions
# ============================================================
try:
    from openwakeword import utils as oww_utils
    if hasattr(oww_utils, "download_models"):
        oww_utils.download_models()
    elif hasattr(oww_utils, "download_assets"):
        oww_utils.download_assets()
except Exception as e:
    print("Skipping explicit model download:", e)


# ============================================================
# Audio constants / globals
# ============================================================
FORMAT = pyaudio.paInt16
CHANNELS = 1
TARGET_RATE = 16000
CHUNK = int(args.chunk_size)

pa: Optional[pyaudio.PyAudio] = None
mic_stream = None
stream_rate = TARGET_RATE
device_rate = None
device_index = None


# ============================================================
# Helpers: devices + resampling
# ============================================================
def list_input_devices(p: pyaudio.PyAudio) -> List[Tuple[int, Dict[str, Any]]]:
    out = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if int(info.get("maxInputChannels", 0)) > 0:
            out.append((i, info))
    return out


def print_input_devices(p: pyaudio.PyAudio) -> None:
    print("=== INPUT DEVICES ===")
    for i, info in list_input_devices(p):
        print(
            i,
            info.get("name"),
            "rate:",
            info.get("defaultSampleRate"),
            "ch:",
            info.get("maxInputChannels"),
        )
    print("=====================")


def pick_input_device_auto(p: pyaudio.PyAudio) -> Tuple[int, int]:
    """
    Windows-safe picker:
    - avoid soundmapper / generic drivers
    - prefer actual mics (microphone/mikrofon/array/surface/realtek/usb/headset)
    """
    candidates = list_input_devices(p)
    if not candidates:
        raise RuntimeError("No input (capture) devices found. Check Windows mic privacy + devices.")

    skip = ("soundmapper", "primärer", "primary", "treiber", "driver", "lautsprecher")
    prefer = ("microphone", "mikrofon", "mic", "array", "surface", "realtek", "usb", "headset")

    scored = []
    for idx, info in candidates:
        name = (info.get("name") or "").lower()
        if any(s in name for s in skip):
            continue
        score = sum(1 for k in prefer if k in name)
        scored.append((score, idx, info))

    if scored:
        scored.sort(reverse=True)
        _, idx, info = scored[0]
        return idx, int(info.get("defaultSampleRate") or TARGET_RATE)

    idx, info = candidates[0]
    return idx, int(info.get("defaultSampleRate") or TARGET_RATE)


def to_16k_linear_int16(x: np.ndarray, src_rate: int) -> np.ndarray:
    """
    Cheap linear resampler to 16 kHz for wakeword use.
    Input/output: int16 mono.
    """
    if src_rate == TARGET_RATE:
        return x
    factor = TARGET_RATE / float(src_rate)
    idxs = np.linspace(0, len(x) - 1, int(len(x) * factor), endpoint=True)
    y = np.interp(idxs, np.arange(len(x)), x.astype(np.float32))
    return np.clip(np.round(y), -32768, 32767).astype(np.int16)


# ============================================================
# Playback helper (Windows + Linux)
# ============================================================
def play_wav(path: str) -> None:
    if sys.platform.startswith("win"):
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME)
    else:
        cmd = ["aplay"]
        if args.playback_device:
            cmd += ["-D", args.playback_device]
        cmd.append(path)
        try:
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"[PLAY warning] {e}")


# ============================================================
# Model loading (robust across versions)
# ============================================================
def first_supported_param(func, candidates) -> Optional[str]:
    try:
        params = inspect.signature(func).parameters
        for c in candidates:
            if c in params:
                return c
    except Exception:
        pass
    return None


def build_model() -> Model:
    model_kwargs: Dict[str, Any] = {}

    backend_param = first_supported_param(Model.__init__, ("inference_framework", "backend", "inference_backend"))
    if backend_param:
        model_kwargs[backend_param] = args.inference_framework

    if args.model_path:
        path_param = first_supported_param(
            Model.__init__,
            ("wakeword_model_paths", "wakeword_models", "wakeword_model_path", "wakeword_models_paths"),
        )
        if path_param:
            model_kwargs[path_param] = [args.model_path]
        else:
            model_kwargs["wakeword_models"] = [args.model_path]

    return Model(**model_kwargs)


owwModel = build_model()

print("[DEBUG] model_path arg:", args.model_path or "(default models)")
try:
    print("[DEBUG] prediction_buffer keys (before predict):", list(getattr(owwModel, "prediction_buffer", {}).keys()))
except Exception:
    pass


# ============================================================
# Score/key extraction (don’t depend on prediction_buffer)
# ============================================================
def preferred_stem() -> Optional[str]:
    if not args.model_path:
        return None
    try:
        return Path(args.model_path).stem
    except Exception:
        return None


def pick_key_from_prediction(prediction: Any, preferred: Optional[str]) -> Optional[str]:
    if not isinstance(prediction, dict) or not prediction:
        return None
    keys = list(prediction.keys())
    if preferred:
        pl = preferred.lower()
        for k in keys:
            if pl in str(k).lower():
                return k
    # nice default if you're testing alexa etc.
    if "alexa" in prediction:
        return "alexa"
    return keys[0]


def extract_score(prediction: Any, key: Optional[str]) -> Optional[float]:
    try:
        if isinstance(prediction, dict):
            if key and key in prediction:
                return float(prediction[key])
            if len(prediction) == 1:
                return float(next(iter(prediction.values())))
            return float(next(iter(prediction.values())))
        return float(prediction)
    except Exception:
        return None


def format_model_label(key: Optional[str], fallback: str) -> str:
    if not key:
        return fallback
    try:
        return Path(str(key)).stem or str(key)
    except Exception:
        return str(key)


# ============================================================
# Peak-picking trigger (Solution B)
# ============================================================
class PeakPicker:
    """
    Arms a trigger when score has fallen below release threshold.
    Fires exactly once per "activation blob" above threshold.
    """
    def __init__(self, threshold: float, release_ratio: float = 0.5):
        self.threshold = float(threshold)
        self.release_ratio = float(release_ratio)
        # Clamp release ratio to sane range
        if not (0.01 <= self.release_ratio <= 0.99):
            self.release_ratio = 0.5
        self.release = self.threshold * self.release_ratio
        self.armed = True

    def update_threshold(self, threshold: float):
        self.threshold = float(threshold)
        self.release = self.threshold * self.release_ratio

    def update_release_ratio(self, release_ratio: float):
        self.release_ratio = float(release_ratio)
        if not (0.01 <= self.release_ratio <= 0.99):
            self.release_ratio = 0.5
        self.release = self.threshold * self.release_ratio

    def step(self, score: float) -> bool:
        """
        Returns True exactly on a peak crossing event (armed & score>=threshold),
        then disarms until score drops below release.
        """
        if self.armed and score >= self.threshold:
            self.armed = False
            return True
        if (not self.armed) and score < self.release:
            self.armed = True
        return False


# ============================================================
# Mic init (live only)
# ============================================================
def init_mic_live() -> None:
    global pa, mic_stream, stream_rate, device_rate, device_index
    pa = pyaudio.PyAudio()

    if args.list_devices:
        print_input_devices(pa)
        raise SystemExit(0)

    if args.mic_device >= 0:
        device_index = int(args.mic_device)
        info = pa.get_device_info_by_index(device_index)
        device_rate = int(info.get("defaultSampleRate") or TARGET_RATE)
    else:
        device_index, device_rate = pick_input_device_auto(pa)

    stream_rate = TARGET_RATE
    try:
        mic_stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=stream_rate,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=device_index,
        )
    except OSError:
        stream_rate = int(device_rate or TARGET_RATE)
        mic_stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=stream_rate,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=device_index,
        )

    print("Using input device index:", device_index)
    print("Device info:", pa.get_device_info_by_index(device_index))
    print("Stream rate:", stream_rate, "-> model rate:", TARGET_RATE)


# ============================================================
# Capture + transcription
# ============================================================
DETECTION_THRESHOLD = float(args.threshold)
RELEASE_RATIO = float(args.release_ratio)

CAPTURE_SECONDS = max(0.0, float(args.capture_seconds))
CAPTURE_SAMPLES_TARGET = int(CAPTURE_SECONDS * TARGET_RATE)

OUTPUT_DIR = Path(args.output_dir).expanduser()
capture_enabled = CAPTURE_SAMPLES_TARGET > 0
if capture_enabled:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRANSCRIBE_URL = args.transcribe_url.strip()
TRANSCRIBE_TIMEOUT = max(0.1, float(args.transcribe_timeout))
transcription_enabled = bool(TRANSCRIBE_URL)


def write_wav_int16_mono(samples_int16: bytes, sample_rate: int, dest: Path) -> None:
    with wave.open(str(dest), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # PCM16
        wf.setframerate(sample_rate)
        wf.writeframes(samples_int16)


def request_transcription(file_path: Path) -> Optional[str]:
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

    try:
        payload = resp.json()
        if isinstance(payload, dict):
            text = payload.get("text") or payload.get("transcription")
            if not text:
                segments = payload.get("segments")
                if isinstance(segments, list):
                    text = " ".join(
                        str(seg.get("text", "")).strip()
                        for seg in segments
                        if isinstance(seg, dict)
                    ).strip()
            return text.strip() if text else None
        return resp.text.strip() or None
    except ValueError:
        return resp.text.strip() or None


# ============================================================
# Offline WAV reading (int16 mono @16k)
# ============================================================
def read_wav_int16_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        raise ValueError(f"{path} is not 16-bit PCM (sampwidth={sampwidth})")

    x = np.frombuffer(raw, dtype=np.int16)

    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
        x = np.clip(np.round(x), -32768, 32767).astype(np.int16)

    if sr != TARGET_RATE:
        x = to_16k_linear_int16(x, sr)

    return x


# ============================================================
# Offline neg runner (now peak-picking, not debounce)
# ============================================================
def run_offline_negative_eval(neg_dir: Path) -> None:
    files = sorted(p for p in neg_dir.rglob("*.wav"))
    if not files:
        raise RuntimeError(f"No WAV files in {neg_dir}")

    hop_samples = max(1, int(TARGET_RATE * float(args.hop_ms) / 1000.0))
    chunk_samples = CHUNK

    total_seconds = 0.0
    false_alarms = 0
    global_max_score = 0.0

    preferred = preferred_stem()
    printed_once = False

    for f_i, wav_path in enumerate(files, start=1):
        print(f"[OFFLINE_NEG] file {f_i}/{len(files)}: {wav_path.name}")

        audio = read_wav_int16_mono(wav_path)
        total_seconds += len(audio) / float(TARGET_RATE)

        picker = PeakPicker(DETECTION_THRESHOLD, RELEASE_RATIO)
        max_score_file = 0.0
        triggers_file = 0

        i = 0
        last_prediction = None

        while i + chunk_samples <= len(audio):
            frame = audio[i:i + chunk_samples]  # int16 mono @ 16k

            prediction = owwModel.predict(frame)
            last_prediction = prediction

            key = pick_key_from_prediction(prediction, preferred)
            score = extract_score(prediction, key)

            if score is not None:
                if score > max_score_file:
                    max_score_file = score

                if picker.step(score):
                    false_alarms += 1
                    triggers_file += 1

            i += hop_samples

        global_max_score = max(global_max_score, max_score_file)

        if not printed_once and last_prediction is not None:
            printed_once = True
            print("[DEBUG] type(prediction):", type(last_prediction))
            if isinstance(last_prediction, dict):
                print("[DEBUG] prediction keys:", list(last_prediction.keys()))
                for k in list(last_prediction.keys())[:10]:
                    try:
                        print("  ", k, "=", float(last_prediction[k]))
                    except Exception:
                        print("  ", k, "=", repr(last_prediction[k])[:120])
            try:
                print("[DEBUG] prediction_buffer keys (after predict):",
                      list(getattr(owwModel, "prediction_buffer", {}).keys()))
            except Exception:
                pass

        print(f"[DEBUG] file_max_score={max_score_file:.6f} triggers_in_file={triggers_file}")

    hours = total_seconds / 3600.0
    far = (false_alarms / hours) if hours > 0 else float("nan")

    print("\n" + "#" * 60)
    print("OFFLINE NEGATIVE EVALUATION (Peak-picking)")
    print(f"Files           : {len(files)}")
    print(f"Audio duration  : {hours:.2f} h")
    print(f"Triggers        : {false_alarms}")
    print(f"FAR             : {far:.6f} FA/h")
    print(f"Threshold       : {DETECTION_THRESHOLD}")
    print(f"Release ratio   : {RELEASE_RATIO}  (release={DETECTION_THRESHOLD * RELEASE_RATIO:.6f})")
    print(f"hop_ms          : {args.hop_ms}")
    print(f"Global maxscore : {global_max_score:.6f}")
    print("#" * 60 + "\n")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    # OFFLINE
    if args.mode == "offline_neg":
        if not args.neg_dir:
            raise RuntimeError("--neg_dir required for offline_neg mode")
        run_offline_negative_eval(Path(args.neg_dir))
        raise SystemExit(0)

    # LIVE
    init_mic_live()

    preferred = preferred_stem()
    print("#" * 60)
    print("Listening for a single wakeword...")
    if preferred:
        print(f"Preferred model: {preferred}")
    print("#" * 60)

    prediction_key = None
    model_label = preferred or "wakeword"

    picker_live = PeakPicker(DETECTION_THRESHOLD, RELEASE_RATIO)

    # Recording state
    recording_active = False
    record_buffer = bytearray()
    record_samples = 0
    record_label: Optional[str] = None
    record_started_at: Optional[datetime] = None

    try:
        voice = PiperVoice.load("./en_US-lessac-medium.onnx")

        el = eliza.Eliza()
        el.load("doctor.txt")

        while True:
            try:
                raw = mic_stream.read(CHUNK, exception_on_overflow=False)
            except OSError as e:
                print(f"[Audio warning] read() failed: {e}. Retrying...")
                continue

            frame = np.frombuffer(raw, dtype=np.int16)

            # resample if device isn't 16k
            if stream_rate != TARGET_RATE:
                frame = to_16k_linear_int16(frame, stream_rate)

            # Keep int16 for predict
            try:
                prediction = owwModel.predict(frame)
            except Exception as e:
                print(f"[OWW warning] predict() failed: {e}. Continuing...")
                continue

            if prediction_key is None:
                prediction_key = pick_key_from_prediction(prediction, preferred)
                model_label = format_model_label(prediction_key, model_label)

            score = extract_score(prediction, prediction_key)
            if score is None:
                continue

            triggered = picker_live.step(score)

            if triggered:
                print(f"[TRIGGER] Wakeword '{model_label}' detected (score={score:.3f})")
                play_wav("./start_listening.wav")

            # Capture: store PCM16 bytes
            if triggered and capture_enabled and not recording_active:
                record_buffer = bytearray(frame.tobytes())
                record_samples = len(frame)
                record_label = model_label
                record_started_at = datetime.now()
                recording_active = True
            elif capture_enabled and recording_active:
                record_buffer.extend(frame.tobytes())
                record_samples += len(frame)

            if capture_enabled and recording_active and record_samples >= CAPTURE_SAMPLES_TARGET:
                timestamp = (record_started_at or datetime.now()).strftime("%Y%m%d-%H%M%S")
                safe_label = "".join(
                    c if c.isalnum() or c in ("-", "_") else "_" for c in (record_label or model_label)
                ).strip("_") or "wakeword"
                dest = Path(args.output_dir).expanduser() / f"{timestamp}_{safe_label}.wav"

                try:
                    write_wav_int16_mono(bytes(record_buffer), TARGET_RATE, dest)
                    print(f"[CAPTURE] Saved {CAPTURE_SECONDS:.2f}s of audio to {dest}")

                    transcription = request_transcription(dest)
                    if transcription:
                        with wave.open("test.wav", "wb") as wav_file:
                            voice.synthesize_wav(transcription, wav_file)
                        play_wav("./test.wav")
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
