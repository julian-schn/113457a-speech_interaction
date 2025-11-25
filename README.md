# 113457a Speech Interaction
[Lecture by M. Heisler](https://hdm-stuttgart.de/vorlesung_detail?vorlid=5215695)

- Lecture notes, code and other information for the "Speech Interaction" lecture at HdM Stuttgart
- Slides and reference [here](https://heisler.pages.mi.hdm-stuttgart.de/si/intro.html)
- Notes and documentation hosted on [GitHub pages](https://julian-schn.github.io/113457a-speech_interaction/) automatically built using a VitePress pipeline

## Basic Wakeword Detection Demo

The folder `code/basic-wakeword detection/` contains a minimal wakeword listener (`app.py`) that streams audio from the default microphone, runs it through a single [OpenWakeWord](https://github.com/dscripka/openwakeword) model and prints a `[TRIGGER]` log once the score crosses the detection threshold.

### Setup

```bash
cd code/basic-wakeword\ detection
source venv/bin/activate
pip install --upgrade pip
pip install --break-system-packages openwakeword
```

The virtual environment (`venv/`) already includes `pyaudio` and other audio dependencies; the extra `pip` command above installs OpenWakeWord and its runtime so the script can load ONNX models locally.

### Example model

An example model is stored in `code/basic-wakeword detection/models/hey_mycroft_v0.1.onnx` (copied from the OpenWakeWord distribution). Use it as-is or drop additional `.onnx` wakeword detectors into the same `models/` directory.

### Running the listener

```bash
python app.py --model_path models/hey_mycroft_v0.1.onnx
```

Optional flags:

- `--chunk_size`: audio frame size in samples (defaults to `1280`)
- `--inference_framework`: backend to use (defaults to `onnx`)
- `--capture_seconds`: seconds of audio to save after each trigger (set `0` to disable)
- `--output_dir`: where captured `.wav` snippets are stored (defaults to `recordings/`)
- `--transcribe_url`: whisper.cpp server endpoint (e.g. `http://127.0.0.1:8080/inference`)
- `--transcribe_timeout`: how long to wait for the transcription response in seconds

When a wakeword is detected, the script prints a debounce-protected line such as:

```
[TRIGGER] Wakeword 'hey_mycroft_v0.1' detected (score=0.812)
```

If `--capture_seconds` is non-zero, the script buffers audio after each trigger and writes it to `output_dir` once the duration elapses:

```
[CAPTURE] Saved 2.00s of audio to recordings/20251110-130000_hey_mycroft_v0.1.wav
```

If `--transcribe_url` is set, each saved snippet is sent to the whisper.cpp REST API and the transcription is logged:

```
[TRANSCRIBE] turn on the lights
```

### Transcribing with whisper.cpp

1. Build and prepare whisper.cpp (one-time setup):
   ```bash
   cd ~/Desktop
   git clone https://github.com/ggerganov/whisper.cpp.git
   cd whisper.cpp
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j$(nproc)
   ./models/download-ggml-model.sh base.en   # or any other model you prefer
   ```
2. Launch the HTTP server in another terminal:
   ```bash
   ./whisper.cpp/build/bin/whisper-server \
     --model ./whisper.cpp/models/ggml-base.en.bin \
     --host 127.0.0.1 \
     --port 8080
   ```
3. Start the wakeword app with transcription enabled:
   ```bash
   python app.py \
     --model_path models/hey_mycroft_v0.1.onnx \
     --capture_seconds 2 \
     --transcribe_url http://127.0.0.1:8080/inference
     --playback_device hw:2,0
   ```
4. After each trigger, the captured `.wav` is uploaded to the Whisper server via `requests` (multipart `file=` payload) and the returned text is printed.

Use these hooks to integrate downstream actions, collect training data, or experiment with other wakeword models.
