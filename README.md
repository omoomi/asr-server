# SenseVoice RKNN Server for RK3588

Multilingual speech recognition server with language detection, emotion recognition, and audio event detection, optimized for Rockchip RK3588 NPU.

## Features

- **Automatic Speech Recognition (ASR)** - Multilingual transcription (Chinese, English, Cantonese, Japanese, Korean)
- **Language Identification (LID)** - Automatic language detection
- **Speech Emotion Recognition (SER)** - Emotion classification (HAPPY, SAD, ANGRY, NEUTRAL, etc.)
- **Audio Event Detection (AED)** - Event classification (Speech, BGM, Applause, Laughter, etc.)
- **NPU Acceleration** - ~15x realtime inference on RK3588

## Requirements

- Rockchip RK3588 (or compatible NPU)
- Python 3.10+
- RKNN runtime library (`librknnrt.so`)

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

### Start Server

```bash
source venv/bin/activate
python server.py
# or: venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
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

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio_file_path` | string | required | Path to audio file (POST /transcribe) |
| `file` | file | required | Audio file upload (POST /transcribe/upload) |
| `language` | string | "auto" | Language: auto, zh, en, yue, ja, ko |
| `use_itn` | bool | false | Apply inverse text normalization |

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
├── download_models.py      # Model download script
├── lib/librknnrt.so        # RKNN runtime library
├── models/                 # Model files (downloaded)
├── samples/                # Test audio files
├── sensevoice/
│   ├── __init__.py
│   ├── inference.py        # RKNN inference wrapper
│   └── postprocess.py      # Output parsing
├── server.py               # FastAPI server
├── requirements.txt
└── README.md
```

## References

- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice)
- [RKNN Toolkit 2](https://github.com/airockchip/rknn-toolkit2)
- [SenseVoiceSmall-RKNN2](https://huggingface.co/happyme531/SenseVoiceSmall-RKNN2)
