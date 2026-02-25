# SenseVoice RKNN Server for RK3588

Multilingual speech recognition server with language detection, emotion recognition, and audio event detection, optimized for Rockchip RK3588 NPU.

## Features

- **Automatic Speech Recognition (ASR)** - Multilingual transcription (Chinese, English, Cantonese, Japanese, Korean)
- **Language Identification (LID)** - Automatic language detection
- **Speech Emotion Recognition (SER)** - Emotion classification (HAPPY, SAD, ANGRY, NEUTRAL, etc.)
- **Audio Event Detection (AED)** - Event classification (Speech, BGM, Applusause, Laughter, etc.)
- **NPU Acceleration** - ~15x realtime inference on RK3588
- **Push-to-Talk CLI** - Record and transcribe from microphone

## Requirements

- Rockchip RK3588 (or compatible NPU)
- Python 3.10+
- RKNN runtime library (`librknnrt.so`)
- Microphone (for push-to-talk mode)

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download models (~485 MB)
python download_models.py

# Install RKNN runtime
sudo cp lib/librknnrt.so /usr/lib/
sudo ldconfig
```

## Usage

### CLI Commands

```bash
# Transcribe an audio file
python cli.py transcribe samples/jfk.wav

# Push-to-talk mode (press Enter to start/stop recording)
python cli.py talk

# Start API server
python cli.py server --port 8000

# List audio input devices
python cli.py list-devices
```

### CLI Options

```
python cli.py [OPTIONS] COMMAND [ARGS]

Options:
  --model-dir PATH    Model directory (default: models)
  --language, -l      Language: auto, zh, en, yue, ja, ko (default: auto)
  --verbose, -v       Enable verbose logging

Commands:
  transcribe FILE     Transcribe an audio file
  talk                Push-to-talk mode (Enter to start/stop)
  server              Start API server
  list-devices        List audio input devices

transcribe options:
  --itn               Apply inverse text normalization
  --raw               Print raw output with tokens

talk options:
  --device, -d INT    Audio device ID
  --max-duration, -m  Max recording duration in seconds (default: 30)

server options:
  --host HOST         Server host (default: 0.0.0.0)
  --port, -p INT      Server port (default: 8000)
```

### API Endpoints

**Health Check**
```bash
curl http://localhost:8000/health
```

**Transcribe by Path**
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"audio_file_path": "samples/jfk.wav"}' \
  http://localhost:8000/transcribe
```

**Upload Audio File**
```bash
curl -X POST -F "file=@audio.wav" \
  http://localhost:8000/transcribe/upload
```

### Response Format

```json
{
  "text": "transcribed text",
  "language": "en",
  "emotion": "NEUTRAL",
  "events": ["Speech"],
  "raw": "<|en|><|NEUTRAL|><|Speech|><|woitn|>transcribed text",
  "segments": [...]
}
```

### Python API

```python
from sensevoice import SenseVoiceRKNN, format_output

model = SenseVoiceRKNN("models")
result = model.transcribe("audio.wav", language="auto")
formatted = format_output(result)

print(f"Text: {formatted['text']}")
print(f"Language: {formatted['language']}")
print(f"Emotion: {formatted['emotion']}")
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SENSEVOICE_MODEL_DIR` | `models` | Model directory path |
| `SENSEVOICE_HOST` | `0.0.0.0` | Server host |
| `SENSEVOICE_PORT` | `8000` | Server port |

## Project Structure

```
asr-server/
├── cli.py                    # CLI with push-to-talk support
├── download_models.py        # Model download script
├── lib/librknnrt.so          # RKNN runtime library
├── models/                   # Model files (downloaded)
├── samples/                  # Test audio files
├── sensevoice/
│   ├── __init__.py
│   ├── inference.py          # RKNN inference wrapper
│   └── postprocess.py        # Output parsing
├── server.py                 # FastAPI server
├── requirements.txt
└── README.md
```

## References

- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice)
- [RKNN Toolkit 2](https://github.com/airockchip/rknn-toolkit2)
- [SenseVoiceSmall-RKNN2](https://huggingface.co/happyme531/SenseVoiceSmall-RKNN2)
