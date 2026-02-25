#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI for ASR with push-to-talk support."""

import argparse
import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("SENSEVOICE_MODEL_DIR", "models")


def check_audio_deps():
    """Check if audio dependencies are installed."""
    try:
        import sounddevice
        return True
    except ImportError:
        return False


def list_audio_devices():
    """List available audio input devices."""
    if not check_audio_deps():
        print("Error: sounddevice not installed. Run: pip install sounddevice")
        return []
    
    import sounddevice as sd
    devices = []
    default_input = sd.default.device[0]
    
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            marker = " [DEFAULT]" if i == default_input else ""
            devices.append(f"  [{i}] {dev['name']}{marker} ({int(dev['default_samplerate'])}Hz)")
    
    return devices


class PushToTalkRecorder:
    """Push-to-talk recorder with Enter key trigger."""
    
    def __init__(self, device=None, sample_rate=16000, max_duration=30.0):
        if not check_audio_deps():
            raise RuntimeError("sounddevice not installed. Run: pip install sounddevice")
        
        import sounddevice as sd
        self.sd = sd
        self.device = device
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self._recording = False
        self._audio_chunks = []
        self._stream = None
    
    def _audio_callback(self, indata, frames, time_info, status):
        if self._recording:
            self._audio_chunks.append(indata.copy())
    
    def start_recording(self):
        """Start recording in background."""
        self._audio_chunks = []
        self._recording = True
        self._stream = self.sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("Recording started")
    
    def stop_recording(self):
        """Stop recording and return audio data."""
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
        
        if not self._audio_chunks:
            return np.array([], dtype=np.float32), self.sample_rate
        
        audio = np.concatenate(self._audio_chunks).flatten()
        logger.info(f"Recording stopped: {len(audio)} samples ({len(audio)/self.sample_rate:.2f}s)")
        return audio, self.sample_rate
    
    def record_push_to_talk(self):
        """Record using Enter key to start/stop."""
        print("Press ENTER to start recording...")
        input()

        self.start_recording()
        print(f"[RECORDING] Press ENTER to stop (max {self.max_duration}s)...")
        
        start_time = time.time()

        def timeout_check():
            while self._recording:
                elapsed = time.time() - start_time
                if elapsed >= self.max_duration:
                    print("\n[TIMEOUT] Max duration reached")
                    self._recording = False
                    break
                time.sleep(0.1)
        
        timeout_thread = threading.Thread(target=timeout_check, daemon=True)
        timeout_thread.start()
        
        input()
        return self.stop_recording()


def cmd_transcribe(args):
    """Transcribe an audio file."""
    from sensevoice import SenseVoiceRKNN, format_output
    
    model = SenseVoiceRKNN(args.model_dir)
    result = model.transcribe(args.file, language=args.language, use_itn=args.itn)
    formatted = format_output(result)
    
    if args.raw:
        print(formatted["raw"])
    else:
        print(f"Language: {formatted['language']}")
        print(f"Emotion: {formatted['emotion']}")
        print(f"Events: {formatted['events']}")
        print(f"Text: {formatted['text']}")
    
    return 0


def get_best_input_device():
    """Get the best input device, preferring USB devices."""
    if not check_audio_deps():
        return None
    
    import sounddevice as sd
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            name_lower = dev["name"].lower()
            if "usb" in name_lower or "brio" in name_lower:
                return i, dev["name"]
    
    if default_input >= 0:
        return default_input, devices[default_input]["name"]
    
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            return i, dev["name"]
    
    return None, None


def cmd_talk(args):
    """Push-to-talk transcription."""
    from sensevoice import SenseVoiceRKNN, format_output
    
    print("ASR Push-to-Talk Mode")
    print("=" * 40)
    
    device = args.device
    device_name = None
    
    if device is None:
        device, device_name = get_best_input_device()
        if device is None:
            print("No input device found!")
            return 1
        print(f"Using device: {device_name}")
    else:
        import sounddevice as sd
        device_name = sd.query_devices(device)["name"]
        print(f"Using device: {device_name}")
    
    model = SenseVoiceRKNN(args.model_dir)
    recorder = PushToTalkRecorder(device=device, max_duration=args.max_duration)
    
    print(f"Max duration: {args.max_duration}s")
    print()
    
    while True:
        try:
            audio, sr = recorder.record_push_to_talk()
            
            if len(audio) == 0:
                print("[No audio recorded]")
                continue
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                import soundfile as sf
                sf.write(tmp.name, audio, sr)
                tmp_path = tmp.name
            
            try:
                result = model.transcribe(tmp_path, language=args.language)
                formatted = format_output(result)
                print()
                print(f"[{formatted['language'] or '??'}] {formatted['text']}")
                print()
            finally:
                Path(tmp_path).unlink()
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
    
    return 0


def cmd_server(args):
    """Start the API server."""
    import subprocess
    subprocess.run([
        sys.executable, "-m", "uvicorn", "server:app",
        "--host", args.host, "--port", str(args.port)
    ])
    return 0


def cmd_list_devices(args):
    """List audio input devices."""
    devices = list_audio_devices()
    
    if not devices:
        print("No audio input devices found")
        return 1
    
    print("Available audio input devices:")
    print("-" * 60)
    for dev in devices:
        print(dev)
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="asr",
        description="ASR - Automatic Speech Recognition for RK3588",
    )
    
    parser.add_argument(
        "--model-dir",
        default=MODEL_DIR,
        help="Model directory path (default: models)",
    )
    parser.add_argument(
        "--language", "-l",
        default="auto",
        choices=["auto", "zh", "en", "yue", "ja", "ko"],
        help="Language code (default: auto)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # transcribe command
    trans_parser = subparsers.add_parser("transcribe", help="Transcribe an audio file")
    trans_parser.add_argument("file", help="Path to audio file")
    trans_parser.add_argument("--itn", action="store_true", help="Apply inverse text normalization")
    trans_parser.add_argument("--raw", action="store_true", help="Print raw output with tokens")
    trans_parser.set_defaults(func=cmd_transcribe)
    
    # talk command
    talk_parser = subparsers.add_parser("talk", help="Push-to-talk mode")
    talk_parser.add_argument("--device", "-d", type=int, help="Audio device ID")
    talk_parser.add_argument("--max-duration", "-m", type=float, default=30.0, help="Max recording duration")
    talk_parser.set_defaults(func=cmd_talk)
    
    # server command
    server_parser = subparsers.add_parser("server", help="Start API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Server host")
    server_parser.add_argument("--port", "-p", type=int, default=8000, help="Server port")
    server_parser.set_defaults(func=cmd_server)
    
    # list-devices command
    list_parser = subparsers.add_parser("list-devices", help="List audio input devices")
    list_parser.set_defaults(func=cmd_list_devices)
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.command is None:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
