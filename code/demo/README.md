# Wakeword Demo for Presentation

A simplified demonstration script showing a complete voice interaction pipeline:
1. Wakeword detection ("How do you wanna do this")
2. Audio capture (VAD-based, stops when user stops speaking)
3. Speech-to-text transcription
4. Keyword-based response with sound effects

## Workflow

```
User speaks wakeword → Acknowledgment sound plays → Record until speech stops (VAD) →
Transcribe with Whisper → Match keywords → Play response sound
```

**Keyword Responses:**
- Contains "**sword**" → plays `sounds/sword.wav`
- Contains "**cast**" → plays `sounds/magic.wav`
- No match → prints message, no sound

## Prerequisites

### 1. Custom Wakeword Model

You need a trained OpenWakeWord model for the phrase "How do you wanna do this".

**Place the model file at:**
```
models/how_do_you_wanna_do_this.onnx
```

**Training a Custom Model:**
- See OpenWakeWord documentation: https://github.com/dscripka/openwakeword
- Use the custom model training guide to create your own wakeword

### 2. Whisper.cpp Server

The demo sends audio to a local whisper.cpp server for transcription.

**Setup (one-time):**
```bash
# Navigate to a temporary location
cd ~/Desktop

# Clone and build whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Download a model (base.en recommended for speed)
./models/download-ggml-model.sh base.en
```

**Start the server (before running demo):**
```bash
cd ~/Desktop/whisper.cpp
./build/bin/whisper-server \
  --model ./models/ggml-base.en.bin \
  --host 127.0.0.1 \
  --port 8080
```

Keep this terminal running while you use the demo.

### 3. Audio Hardware

- **Microphone**: USB microphone recommended (auto-detected)
- **Speakers/Headphones**: For sound playback via `aplay`
- **Linux required**: Uses ALSA (`aplay` command)

## Setup

### 1. Create Virtual Environment

```bash
cd code/demo
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Add Your Custom Model

Place your trained wakeword model:
```bash
# Copy your custom model to:
cp /path/to/your/model.onnx models/how_do_you_wanna_do_this.onnx
```

### 4. Customize Sound Effects (Optional)

The demo includes placeholder sounds copied from `start_listening.wav`. Replace them with custom sounds:

```bash
# Replace with your own WAV files
cp /path/to/your/acknowledgment.wav sounds/acknowledgment.wav
cp /path/to/your/sword_sound.wav sounds/sword.wav
cp /path/to/your/magic_sound.wav sounds/magic.wav
```

## Running the Demo

### 1. Start Whisper.cpp Server

In a separate terminal:
```bash
cd ~/Desktop/whisper.cpp
./build/bin/whisper-server \
  --model ./models/ggml-base.en.bin \
  --host 127.0.0.1 \
  --port 8080
```

### 2. Run the Demo

```bash
cd code/demo
source venv/bin/activate
python demo.py
```

### 3. Test It

1. Wait for "Demo Ready! Listening for wakeword..."
2. Say: **"How do you wanna do this"**
3. Listen for acknowledgment sound (beep)
4. Speak your command (recording stops automatically when you finish):
   - "**With my sword**" → sword sound plays
   - "**I cast fireball**" → magic sound plays
5. The system uses VAD to detect when you stop speaking
6. Repeat!

## Command-Line Options

```bash
# Use a different model
python demo.py --model_path models/alternative_model.onnx

# Use a different Whisper endpoint
python demo.py --transcribe_url http://192.168.1.100:8080/inference
```

## Troubleshooting

### "Model not found" Error
- Ensure your custom model is at `models/how_do_you_wanna_do_this.onnx`
- For testing, you can use an existing model:
  ```bash
  cp ../basic-wakeword\ detection/models/hey_mycroft_v0.1.onnx \
     models/how_do_you_wanna_do_this.onnx
  ```

### "No input devices found"
- Check microphone connection: `arecord -l`
- Try: `sudo apt install alsa-utils` (Linux)

### Transcription Fails
- Verify whisper.cpp server is running
- Check endpoint: `curl http://127.0.0.1:8080/`
- Ensure port 8080 is not blocked

### No Sound Playback
- Verify `aplay` is installed: `which aplay`
- Test playback: `aplay sounds/acknowledgment.wav`
- Check speaker connection and volume

### Low Wakeword Detection
- Speak clearly and close to microphone
- Adjust threshold in code (line 31): `WAKEWORD_THRESHOLD = 0.3`
- Retrain model with more diverse samples

### Recording Stops Too Early/Late
- Adjust VAD threshold in code (line 33): `VAD_THRESHOLD = 0.3` (lower = more sensitive)
- Increase minimum duration (line 34): `MIN_RECORDING_DURATION = 2.0`
- Check that you're speaking continuously without long pauses

## Code Structure

```
demo.py (~360 lines, well-commented)
├── Configuration (lines 18-43)
│   ├── Wakeword threshold
│   ├── VAD threshold (0.5)
│   ├── Min/max recording duration
│   └── Transcribe endpoint
├── Helper Functions
│   ├── Audio device selection
│   ├── Resampling
│   ├── VAD score extraction
│   ├── WAV file I/O
│   ├── Sound playback
│   ├── Transcription request
│   └── Keyword matching
└── Main Loop
    ├── Initialize PyAudio & Model (with VAD)
    ├── Read audio chunks
    ├── Wakeword detection with debounce
    ├── VAD-based recording (1.5s-10s)
    └── Transcription & response
```

## Differences from `eliza-speech.py`

**Removed:**
- ❌ ELIZA chatbot integration
- ❌ Piper TTS synthesis
- ❌ Multi-model support
- ❌ Complex CLI argument parsing
- ❌ Recording directory management

**Simplified:**
- ✅ Single wakeword model only
- ✅ VAD-based recording (stops when speech ends)
- ✅ Simple keyword matching (no AI chatbot)
- ✅ Direct sound playback (no TTS generation)
- ✅ ~360 lines vs ~470 lines

## For Presentation

This demo is designed for clear demonstration of:
1. **Wakeword Detection** - OpenWakeWord integration
2. **Audio Capture** - PyAudio streaming and buffering
3. **Speech Recognition** - Whisper.cpp HTTP API
4. **Conditional Logic** - Keyword-based response routing

The code prioritizes readability and clarity over robustness, making it easy to walk through during presentations.

## Resources

- **OpenWakeWord**: https://github.com/dscripka/openwakeword
- **Whisper.cpp**: https://github.com/ggerganov/whisper.cpp
- **PyAudio Docs**: https://people.csail.mit.edu/hubert/pyaudio/docs/
