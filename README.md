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

When a wakeword is detected, the script prints a debounce-protected line such as:

```
[TRIGGER] Wakeword 'hey_mycroft_v0.1' detected (score=0.812)
```

Use this hook to integrate downstream actions or to experiment with other wakeword models.
