# -*- coding: utf-8 -*-
"""FastAPI server for SenseVoice RKNN inference on RK3588."""

import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from sensevoice.inference import SenseVoiceRKNN
from sensevoice.postprocess import format_output

logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("SENSEVOICE_MODEL_DIR", "models")
HOST = os.environ.get("SENSEVOICE_HOST", "0.0.0.0")
PORT = int(os.environ.get("SENSEVOICE_PORT", "8000"))

model: Optional[SenseVoiceRKNN] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info(f"Loading SenseVoice RKNN model from {MODEL_DIR}")
    start = time.time()
    model = SenseVoiceRKNN(MODEL_DIR)
    logger.info(f"Model loaded in {time.time() - start:.2f}s")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="SenseVoice RKNN Server",
    description="Multilingual speech recognition, language detection, emotion recognition, and audio event detection on RK3588 NPU",
    version="1.0.0",
    lifespan=lifespan,
)


class TranscribeRequest(BaseModel):
    audio_file_path: str
    language: Literal["auto", "zh", "en", "yue", "ja", "ko"] = "auto"
    use_itn: bool = False


class TranscribeResponse(BaseModel):
    text: str
    language: Optional[str] = None
    emotion: Optional[str] = None
    events: list[str] = []
    raw: str = ""
    segments: list[dict] = []


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "loading",
        "model_loaded": model is not None,
        "model_dir": MODEL_DIR,
    }


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_file(request: TranscribeRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    audio_path = Path(request.audio_file_path)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {request.audio_file_path}")

    try:
        logger.info(f"Transcribing: {audio_path}")
        start = time.time()
        result = model.transcribe(
            str(audio_path),
            language=request.language,
            use_itn=request.use_itn,
        )
        logger.info(f"Transcription completed in {time.time() - start:.2f}s")
        return TranscribeResponse(**format_output(result))
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(e))


ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


@app.post("/transcribe/upload", response_model=TranscribeResponse)
async def transcribe_upload(
    file: UploadFile = File(...),
    language: Literal["auto", "zh", "en", "yue", "ja", "ko"] = "auto",
    use_itn: bool = False,
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {extension}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        logger.info(f"Transcribing uploaded file: {file.filename}")
        start = time.time()
        result = model.transcribe(tmp_path, language=language, use_itn=use_itn)
        logger.info(f"Transcription completed in {time.time() - start:.2f}s")
        return TranscribeResponse(**format_output(result))
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "uvicorn", "server:app", "--host", HOST, "--port", str(PORT)])
