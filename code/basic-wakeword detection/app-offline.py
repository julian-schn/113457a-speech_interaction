import argparse
import inspect
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
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
    choices=["live", "offline"],
    help="Run mode: live mic or offline WAV scanning"
)

parser.add_argument(
    "--audio_dir",
    type=str,
    default="",
    help="Directory with WAV files (offline mode)."
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
    default=0.9,
    help="Peak-picking release ratio: next trigger is armed when score < threshold*release_ratio"
)

parser.add_argument(
    "--hop_ms",
    type=float,
    default=40.0,
    help="Hop size in ms for offline audio scanning"
)

parser.add_argument(
    "--chunk_size",
    help="How many samples to predict on at once",
    type=int,
    default=1280,
)

parser.add_argument(
    "--model_path",
    help="Path of a specific model to load (e.g., ./models/hey_mycroft.onnx). If empty, loads default models.",
    type=str,
    default="",
)

parser.add_argument(
    "--inference_framework",
    help="Inference backend to use (onnx/tflite). Some versions may ignore this.",
    type=str,
    default="onnx",
)

parser.add_argument(
    "--capture_seconds",
    help="How long after a trigger to keep recording audio before saving (0 disables saving)",
    type=float,
    default=2.0,
)

parser.add_argument(
    "--output_dir",
    help="Directory to store captured wav files",
    type=str,
    default="recordings",
)

parser.add_argument(
    "--transcribe_url",
    help="Whisper.cpp inference endpoint (e.g. http://127.0.0.1:8080/inference). Leave empty to skip transcription.",
    type=str,
    default="",
)

parser.add_argument(
    "--transcribe_timeout",
    help="Seconds to wait for Whisper.cpp transcription responses",
    type=float,
    default=30.0,
)

parser.add_argument(
    "--playback_device",
    help="ALSA device string for aplay (Linux). Leave empty to use default device.",
    type=str,
    default="",
)

parser.add_argument(
    "--mic_device",
    help="PyAudio input device index. -1 = auto pick.",
    type=int,
    default=-1,
)

parser.add_argument(
    "--list_devices",
    help="Print input devices and exit (live mode only)",
    action="store_true",
)

parser.add_argument(
    "--play_trigger_sound",
    action="store_true",
    help="Play ./start_listening.wav on trigger (works on Windows+Linux)."
)

parser.add_argument(
    "--print_triggers",
    action="store_true",
    help="Offline: collect trigger timestamps across ALL files; Live: print trigger clock time."
)

parser.add_argument(
    "--debug_once",
    action="store_true",
    help="Print one-time debug info about prediction output (offline + live)."
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
FORMAT = None          # set in live mode after pyaudio import
CHANNELS = 1
TARGET_RATE = 16000
CHUNK = int(args.chunk_size)

pyaudio = None         # lazy import for live mode only
pa = None
mic_stream = None
stream_rate = TARGET_RATE
device_rate = None
device_index = None


# ============================================================
# Helpers: resampling
# ============================================================
def to_16k_linear_int16(x: np.ndarray, src_rate: int) -> np.ndarray:
    """Cheap linear resampler to 16 kHz for wakeword use. Input/output: int16 mono."""
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
    if not args.play_trigger_sound:
        return
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


# ============================================================
# Score/key extraction
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
# Peak-picking trigger
# ============================================================
class PeakPicker:
    """Fires once per 'activation blob' above threshold, rearms when score < threshold*release_ratio."""
    def __init__(self, threshold: float, release_ratio: float = 0.9):
        self.threshold = float(threshold)
        self.release_ratio = float(release_ratio)
        if not (0.01 <= self.release_ratio <= 0.99):
            self.release_ratio = 0.9
        self.release = self.threshold * self.release_ratio
        self.armed = True

    def step(self, score: float) -> bool:
        if self.armed and score >= self.threshold:
            self.armed = False
            return True
        if (not self.armed) and score < self.release:
            self.armed = True
        return False


# ============================================================
# Live: devices (lazy pyaudio import)
# ============================================================
def list_input_devices(p) -> List[Tuple[int, Dict[str, Any]]]:
    out = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if int(info.get("maxInputChannels", 0)) > 0:
            out.append((i, info))
    return out


def print_input_devices(p) -> None:
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


def pick_input_device_auto(p) -> Tuple[int, int]:
    candidates = list_input_devices(p)
    if not candidates:
        raise RuntimeError("No input devices found. Check mic privacy + devices.")

    skip = ("soundmapper", "primärer", "primary", "treiber", "driver", "lautsprecher")
    prefer = ("microphone", "mikrofon", "mic", "array", "surface", "realtek", "usb", "headset")

    scored = []
    for idx_, info in candidates:
        name = (info.get("name") or "").lower()
        if any(s in name for s in skip):
            continue
        score = sum(1 for k in prefer if k in name)
        scored.append((score, idx_, info))

    if scored:
        scored.sort(reverse=True)
        _, idx_, info = scored[0]
        return idx_, int(info.get("defaultSampleRate") or TARGET_RATE)

    idx_, info = candidates[0]
    return idx_, int(info.get("defaultSampleRate") or TARGET_RATE)


def init_mic_live() -> None:
    global pyaudio, pa, mic_stream, stream_rate, device_rate, device_index, FORMAT

    import pyaudio as _pyaudio
    pyaudio = _pyaudio

    FORMAT = pyaudio.paInt16
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
# Capture + transcription (live)
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
# Offline runner: ONLY summary at the end
# ============================================================
def run_offline_trigger_count(audio_dir: Path) -> None:
    files = sorted(p for p in audio_dir.rglob("*.wav"))
    if not files:
        raise RuntimeError(f"No WAV files in {audio_dir}")

    hop_samples = max(1, int(TARGET_RATE * float(args.hop_ms) / 1000.0))
    chunk_samples = CHUNK

    preferred = preferred_stem()

    total_triggers = 0
    total_seconds = 0.0
    global_max_score = 0.0

    # Optional: collect ALL trigger timestamps (file-relative seconds) across all files
    # Format: "filename@12.34s"
    all_trigger_marks: List[str] = []

    printed_debug = False

    for wav_path in files:
        audio = read_wav_int16_mono(wav_path)
        total_seconds += len(audio) / float(TARGET_RATE)

        picker = PeakPicker(args.threshold, args.release_ratio)

        i = 0
        last_prediction = None

        while i + chunk_samples <= len(audio):
            frame = audio[i:i + chunk_samples]
            prediction = owwModel.predict(frame)
            last_prediction = prediction

            key = pick_key_from_prediction(prediction, preferred)
            score = extract_score(prediction, key)

            if score is not None:
                if score > global_max_score:
                    global_max_score = score

                if picker.step(score):
                    total_triggers += 1
                    if args.print_triggers:
                        t_s = i / float(TARGET_RATE)
                        all_trigger_marks.append(f"{wav_path.name}@{t_s:.2f}s")

            i += hop_samples

        if args.debug_once and (not printed_debug) and last_prediction is not None:
            printed_debug = True
            print("[DEBUG] model_path arg:", args.model_path or "(default models)")
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
            print()

    print("\n" + "#" * 60)
    print("OFFLINE TRIGGER COUNT (Peak-picking)")
    print(f"Dir            : {audio_dir}")
    print(f"Files          : {len(files)}")
    print(f"Total duration : {total_seconds:.2f} s  ({total_seconds/3600.0:.4f} h)")
    print(f"Triggers total : {total_triggers}")
    print(f"Threshold      : {args.threshold}")
    print(f"Release ratio  : {args.release_ratio}  (release={args.threshold * args.release_ratio:.6f})")
    print(f"hop_ms         : {args.hop_ms}")
    print(f"chunk_samples  : {CHUNK}")
    print(f"Global maxscore: {global_max_score:.6f}")
    if args.print_triggers:
        print(f"Trigger marks  : {len(all_trigger_marks)}")
        if all_trigger_marks:
            # print in one line if small, otherwise wrap a bit
            joined = ", ".join(all_trigger_marks)
            if len(joined) <= 800:
                print("Marks          :", joined)
            else:
                print("Marks          :")
                for m in all_trigger_marks[:200]:
                    print("  ", m)
                if len(all_trigger_marks) > 200:
                    print(f"  ... ({len(all_trigger_marks) - 200} more)")
    print("#" * 60 + "\n")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    # OFFLINE
    if args.mode == "offline":
        if not args.audio_dir:
            raise RuntimeError("--audio_dir required for offline mode")
        run_offline_trigger_count(Path(args.audio_dir))
        raise SystemExit(0)

    # LIVE
    init_mic_live()

    preferred = preferred_stem()
    print("#" * 60)
    print("Listening for wakeword...")
    if preferred:
        print(f"Preferred model: {preferred}")
    print(f"Threshold     : {args.threshold}")
    print(f"Release ratio : {args.release_ratio} (release={args.threshold * args.release_ratio:.6f})")
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

    trigger_counter = 0
    printed_debug_live = False

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

            if stream_rate != TARGET_RATE:
                frame = to_16k_linear_int16(frame, stream_rate)

            try:
                prediction = owwModel.predict(frame)
            except Exception as e:
                print(f"[OWW warning] predict() failed: {e}. Continuing...")
                continue

            if args.debug_once and (not printed_debug_live):
                printed_debug_live = True
                print("[DEBUG] model_path arg:", args.model_path or "(default models)")
                print("[DEBUG] type(prediction):", type(prediction))
                if isinstance(prediction, dict):
                    print("[DEBUG] prediction keys:", list(prediction.keys()))
                print()

            if prediction_key is None:
                prediction_key = pick_key_from_prediction(prediction, preferred)
                model_label = format_model_label(prediction_key, model_label)

            score = extract_score(prediction, prediction_key)
            if score is None:
                continue

            triggered = picker_live.step(score)

            if triggered:
                trigger_counter += 1
                if args.print_triggers:
                    now = datetime.now().strftime("%H:%M:%S")
                    print(f"[TRIGGER #{trigger_counter}] {now}  '{model_label}' score={score:.3f}")
                else:
                    print(f"[TRIGGER] '{model_label}' score={score:.3f}  (count={trigger_counter})")

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
