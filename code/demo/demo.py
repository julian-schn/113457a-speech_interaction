#!/usr/bin/env python3
"""
Simplified Wakeword Demo for Presentation
==========================================
Detects custom wakeword "How do you wanna do this", records audio,
transcribes it, and plays sound effects based on keywords.

Workflow:
  1. Listen for wakeword "How do you wanna do this"
  2. Play acknowledgment sound
  3. Record audio until user stops speaking (VAD-based)
  4. Transcribe with whisper.cpp
  5. Match keywords:
     - "sword" → play sword sound
     - "cast" → play magic sound
"""

import argparse
import time
import wave
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudio
import requests
from openwakeword.model import Model

# ========== Configuration ==========
MODEL_PATH = "models/how_do_you_wanna_do_this.onnx"
WAKEWORD_THRESHOLD = 0.5
DEBOUNCE_SECONDS = 1.0
VAD_THRESHOLD = 0.5  # Voice activity detection threshold
MIN_RECORDING_DURATION = 1.5  # Minimum seconds to record
MAX_RECORDING_DURATION = 10.0  # Maximum seconds to record
TRANSCRIBE_URL = "http://127.0.0.1:8080/inference"
TRANSCRIBE_TIMEOUT = 30.0

# Audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
TARGET_RATE = 16000
CHUNK = 1280  # ~80ms at 16kHz

# ========== Helper Functions ==========

def list_input_devices(pa):
    """Return list of (index, info) for input-capable devices."""
    devices = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if int(info.get("maxInputChannels", 0)) > 0:
            devices.append((i, info))
    return devices


def pick_input_device(pa):
    """Prefer USB mic if available, else first input device."""
    candidates = list_input_devices(pa)
    if not candidates:
        raise RuntimeError("No input devices found. Check your microphone connection.")

    # Look for USB mic
    for i, info in candidates:
        name = (info.get("name") or "").lower()
        if "usb" in name:
            rate = int(info.get("defaultSampleRate") or TARGET_RATE)
            print(f"[AUDIO] Using USB mic: {info.get('name')}")
            return i, rate

    # Fall back to first device
    i, info = candidates[0]
    rate = int(info.get("defaultSampleRate") or TARGET_RATE)
    print(f"[AUDIO] Using default mic: {info.get('name')}")
    return i, rate


def to_16k(audio: np.ndarray, src_rate: int) -> np.ndarray:
    """Resample audio to 16kHz using linear interpolation."""
    if src_rate == TARGET_RATE:
        return audio
    factor = TARGET_RATE / float(src_rate)
    new_length = int(len(audio) * factor)
    indices = np.linspace(0, len(audio) - 1, new_length, endpoint=True)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.int16)


def write_wav(samples: bytes, sample_rate: int, path: Path, pa_instance):
    """Write audio buffer to WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa_instance.get_sample_size(FORMAT))
        wf.setframerate(sample_rate)
        wf.writeframes(samples)


def play_sound(path: str):
    """Play WAV file using aplay."""
    try:
        subprocess.run(["aplay", path], check=False, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print(f"[WARNING] aplay not found. Cannot play {path}")


def request_transcription(file_path: Path) -> Optional[str]:
    """Send WAV file to whisper.cpp and return transcription."""
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

        # Parse response
        payload = resp.json()
        if isinstance(payload, dict):
            text = payload.get("text") or payload.get("transcription")
            if text:
                return str(text).strip()

            # Fallback: join segments
            segments = payload.get("segments")
            if isinstance(segments, list):
                text = " ".join(
                    str(seg.get("text", "")).strip()
                    for seg in segments
                    if isinstance(seg, dict)
                ).strip()
                return text if text else None

        return None

    except requests.RequestException as e:
        print(f"[ERROR] Transcription failed: {e}")
        return None


def get_vad_score(model) -> float:
    """Extract current VAD score from OpenWakeWord model."""
    vad = getattr(model, "vad", None)
    if not vad or not hasattr(vad, "prediction_buffer"):
        return 0.0
    try:
        # Get recent VAD frames and return max score
        vad_frames = list(vad.prediction_buffer)[-20:]
        if not vad_frames:
            return 0.0
        return float(np.max(vad_frames))
    except Exception:
        return 0.0


def handle_transcription(text: str):
    """Match transcription with keywords and play appropriate sound."""
    if not text:
        print("[NO MATCH] Empty transcription")
        return

    print(f"[TRANSCRIPTION] {text}")
    text_lower = text.lower().strip()

    if "sword" in text_lower:
        print("[MATCH] Detected 'sword' → playing sword sound")
        play_sound("sounds/sword.wav")
    elif "cast" in text_lower:
        print("[MATCH] Detected 'cast' → playing magic sound")
        play_sound("sounds/magic.wav")
    else:
        print(f"[NO MATCH] No keyword found in: {text}")


# ========== Main Function ==========

def main():
    parser = argparse.ArgumentParser(description="Wakeword Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help="Path to custom wakeword ONNX model"
    )
    parser.add_argument(
        "--transcribe_url",
        type=str,
        default=TRANSCRIBE_URL,
        help="Whisper.cpp endpoint URL"
    )
    args = parser.parse_args()

    # Validate model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("Please place your custom wakeword model at models/how_do_you_wanna_do_this.onnx")
        return

    # Initialize PyAudio
    pa = pyaudio.PyAudio()

    try:
        # Select and open microphone
        device_idx, device_rate = pick_input_device(pa)

        # Try 16kHz first, fall back to device rate
        stream_rate = TARGET_RATE
        try:
            mic_stream = pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=TARGET_RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=device_idx,
            )
        except OSError:
            stream_rate = device_rate
            mic_stream = pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=stream_rate,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=device_idx,
            )
            print(f"[AUDIO] Using device rate: {stream_rate} Hz")

        # Load wakeword model with VAD enabled
        print(f"[MODEL] Loading: {model_path.name}")
        oww_model = Model(wakeword_models=[str(model_path)], vad_threshold=VAD_THRESHOLD)

        # Setup debounce
        frames_per_second = TARGET_RATE / CHUNK
        debounce_frames = max(1, int(DEBOUNCE_SECONDS * frames_per_second))
        cooldown_remaining = 0

        # Recording state
        recording_active = False
        record_buffer = bytearray()
        record_start_time = None

        print("=" * 60)
        print("Demo Ready! Listening for wakeword...")
        print("Say: 'How do you wanna do this'")
        print("=" * 60)

        # Main loop
        while True:
            # Read audio chunk
            try:
                raw = mic_stream.read(CHUNK, exception_on_overflow=False)
            except OSError as e:
                print(f"[WARNING] Audio read failed: {e}")
                continue

            frame = np.frombuffer(raw, dtype=np.int16)

            # Resample if needed
            if stream_rate != TARGET_RATE:
                frame = to_16k(frame, stream_rate)
            frame_bytes = frame.tobytes()

            # Run wakeword detection
            try:
                prediction = oww_model.predict(frame)
            except Exception as e:
                print(f"[WARNING] Prediction failed: {e}")
                continue

            # Extract score (handle different oww versions)
            score = 0.0
            if hasattr(oww_model, 'prediction_buffer') and oww_model.prediction_buffer:
                key = list(oww_model.prediction_buffer.keys())[0]
                buffer = oww_model.prediction_buffer[key]
                if buffer:
                    score = float(list(buffer)[-1])

            # Update cooldown
            if cooldown_remaining > 0:
                cooldown_remaining -= 1

            # Check for wakeword trigger
            triggered = score > WAKEWORD_THRESHOLD and cooldown_remaining == 0

            if triggered and not recording_active:
                print(f"[TRIGGER] Wakeword detected! (score={score:.3f})")

                # Play acknowledgment sound
                play_sound("sounds/acknowledgment.wav")

                # Start recording
                record_buffer = bytearray(frame_bytes)
                record_start_time = time.perf_counter()
                recording_active = True
                cooldown_remaining = debounce_frames

            elif recording_active:
                # Continue recording
                record_buffer.extend(frame_bytes)
                elapsed = time.perf_counter() - record_start_time

                # Get VAD score to detect when user stops speaking
                vad_score = get_vad_score(oww_model)

                # Determine if we should stop recording:
                # - Must record at least MIN_RECORDING_DURATION seconds
                # - Stop if: VAD drops below threshold (user stopped speaking)
                #   OR max duration reached
                should_stop = False

                if elapsed >= MAX_RECORDING_DURATION:
                    print(f"[CAPTURE] Max duration reached ({elapsed:.2f}s)")
                    should_stop = True
                elif elapsed >= MIN_RECORDING_DURATION and vad_score < VAD_THRESHOLD:
                    print(f"[CAPTURE] Speech ended (VAD={vad_score:.2f}, {elapsed:.2f}s)")
                    should_stop = True

                if should_stop:
                    # Save to file
                    temp_file = Path("temp_recording.wav")
                    write_wav(bytes(record_buffer), TARGET_RATE, temp_file, pa)

                    # Transcribe
                    print("[TRANSCRIBE] Sending to whisper.cpp...")
                    transcription = request_transcription(temp_file)

                    # Handle response
                    if transcription:
                        handle_transcription(transcription)
                    else:
                        print("[ERROR] No transcription received")

                    # Cleanup
                    temp_file.unlink(missing_ok=True)
                    recording_active = False
                    record_buffer = bytearray()
                    record_start_time = None

                    print("\n" + "=" * 60)
                    print("Ready for next wakeword...")
                    print("=" * 60)

    except KeyboardInterrupt:
        print("\n[EXIT] Stopping demo...")

    finally:
        # Cleanup
        try:
            mic_stream.stop_stream()
            mic_stream.close()
        except:
            pass
        pa.terminate()
        print("[EXIT] Cleanup complete")


if __name__ == "__main__":
    main()
