#!/usr/bin/env python3
"""Download SenseVoice RKNN model files from HuggingFace."""

import argparse
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: huggingface_hub not installed")
    print("Install with: pip install huggingface_hub")
    exit(1)

MODEL_ID = "happyme531/SenseVoiceSmall-RKNN2"

MODEL_FILES = [
    ("sense-voice-encoder.rknn", "484 MB"),
    ("chn_jpn_yue_eng_ko_spectok.bpe.model", "377 KB"),
    ("embedding.npy", "36 KB"),
    ("am.mvn", "11 KB"),
    ("fsmn-am.mvn", "8 KB"),
    ("fsmn-config.yaml", "1 KB"),
    ("fsmnvad-offline.onnx", "1.7 MB"),
]


def download_models(output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading SenseVoice RKNN models")
    print(f"Source: {MODEL_ID}")
    print(f"Target: {output_dir}\n")

    for filename, size in MODEL_FILES:
        filepath = output_dir / filename
        if filepath.exists():
            print(f"[SKIP] {filename} ({size})")
            continue
        print(f"[DOWN] {filename} ({size})...")
        hf_hub_download(
            repo_id=MODEL_ID,
            filename=filename,
            local_dir=str(output_dir),
        )

    print("\nDone! Total: ~485 MB")


def main():
    parser = argparse.ArgumentParser(description="Download SenseVoice RKNN models")
    parser.add_argument(
        "-o", "--output",
        default="models",
        help="Output directory (default: models)"
    )
    args = parser.parse_args()
    download_models(args.output)


if __name__ == "__main__":
    main()
