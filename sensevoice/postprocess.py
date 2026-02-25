# -*- coding: utf-8 -*-
"""Postprocessor for SenseVoice model output."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TranscriptionResult:
    text: str
    language: Optional[str] = None
    emotion: Optional[str] = None
    events: List[str] = field(default_factory=list)
    use_itn: bool = False
    raw: str = ""


LANGUAGE_TOKENS = {
    "<|zh|>": "zh",
    "<|en|>": "en",
    "<|yue|>": "yue",
    "<|ja|>": "ja",
    "<|ko|>": "ko",
    "<|nospeech|>": "nospeech",
}

EMOTION_TOKENS = {
    "<|HAPPY|>": "HAPPY",
    "<|SAD|>": "SAD",
    "<|ANGRY|>": "ANGRY",
    "<|NEUTRAL|>": "NEUTRAL",
    "<|FEARFUL|>": "FEARFUL",
    "<|DISGUSTED|>": "DISGUSTED",
    "<|SURPRISED|>": "SURPRISED",
    "<|EMO_UNKNOWN|>": "EMO_UNKNOWN",
}

EVENT_TOKENS = {
    "<|Speech|>": "Speech",
    "<|BGM|>": "BGM",
    "<|Applause|>": "Applause",
    "<|Laughter|>": "Laughter",
    "<|Cry|>": "Cry",
    "<|Sneeze|>": "Sneeze",
    "<|Breath|>": "Breath",
    "<|Cough|>": "Cough",
}

ITN_TOKENS = {
    "<|withitn|>": True,
    "<|woitn|>": False,
}

ALL_TOKENS = {**LANGUAGE_TOKENS, **EMOTION_TOKENS, **EVENT_TOKENS, **ITN_TOKENS}


def parse_transcription(raw_text: str) -> TranscriptionResult:
    text = raw_text
    language = None
    emotion = None
    events = []
    use_itn = False

    for token, lang in LANGUAGE_TOKENS.items():
        if token in text:
            language = lang
            text = text.replace(token, "")
            break

    for token, emo in EMOTION_TOKENS.items():
        if token in text:
            emotion = emo
            text = text.replace(token, "")
            break

    for token, evt in EVENT_TOKENS.items():
        if token in text:
            events.append(evt)
            text = text.replace(token, "")

    for token, itn in ITN_TOKENS.items():
        if token in text:
            use_itn = itn
            text = text.replace(token, "")
            break

    return TranscriptionResult(
        text=text.strip(),
        language=language,
        emotion=emotion,
        events=events,
        use_itn=use_itn,
        raw=raw_text,
    )


def parse_segments(segments: List[dict]) -> List[dict]:
    parsed = []
    for seg in segments:
        result = parse_transcription(seg["text"])
        parsed.append({
            "start_time_s": seg["start_time_s"],
            "end_time_s": seg["end_time_s"],
            "text": result.text,
            "language": result.language,
            "emotion": result.emotion,
            "events": result.events,
            "raw": result.raw,
        })
    return parsed


def format_output(transcription_result: dict) -> dict:
    segments = transcription_result.get("segments", [])
    parsed_segments = parse_segments(segments)

    full_text = transcription_result.get("full_transcription", "")
    parsed_full = parse_transcription(full_text)

    return {
        "text": parsed_full.text,
        "language": parsed_full.language,
        "emotion": parsed_full.emotion,
        "events": parsed_full.events,
        "raw": full_text,
        "segments": parsed_segments,
    }
