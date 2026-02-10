from __future__ import annotations

import os
import sys
import warnings
import argparse
from pathlib import Path

import torch
import whisper
import requests
from dotenv import load_dotenv

from pipeline import analyze_file
from diarization.nemo_config import load_nemo_diar_base_cfg
from utils import parse_user_datetime, parse_filename, should_process

load_dotenv()


HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

DEFAULT_INPUT_DIR = (
    Path("/app/audio_files")
    if Path("/app/audio_files").exists()
    else Path(__file__).resolve().parent.parent / "audios"
)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
AGENT_API = os.getenv("LLM_AGENT_HOST", "http://localhost:5001") + "/analyze"


warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Process audio files in a folder and send to LLM agent")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Folder with audio files")
    parser.add_argument("--start", default=None, help="Start of period (YYYY-MM-DD or ISO datetime)")
    parser.add_argument("--end", default=None, help="End of period (YYYY-MM-DD or ISO datetime)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}")
        sys.exit(1)

    start_dt = parse_user_datetime(args.start, is_end=False) if args.start else None
    end_dt = parse_user_datetime(args.end, is_end=True) if args.end else None

    files = sorted(input_dir.glob("*.mp3"))
    if not files:
        print(f"No .mp3 files found in {input_dir}")
        sys.exit(1)

    print("Loading Whisper model...")
    whisper_model = whisper.load_model(WHISPER_MODEL)

    print("Loading NeMo diarization config...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device.upper()} for NeMo.")
    diar_base_cfg, diar_yaml_path = load_nemo_diar_base_cfg(device=device, num_speakers=2)

    had_errors = False
    processed = 0

    for file_path in files:
        parsed = parse_filename(file_path)
        if not parsed:
            print(f"Skipping file with invalid name format: {file_path.name}")
            continue

        full_name, file_dt, record_time = parsed
        if not should_process(file_dt, start_dt, end_dt):
            continue

        processed += 1
        try:
            ok = analyze_file(
                file_path,
                whisper_model,
                full_name,
                record_time,
                AGENT_API,
                diar_base_cfg,
                diar_yaml_path,
            )
            if not ok:
                had_errors = True
        except requests.exceptions.ConnectionError:
            print(f"Error: Cannot connect to LLM Agent at {AGENT_API}")
            print("Make sure the LLM Agent service is running.")
            had_errors = True
        except Exception as e:
            print(f"Failed to process {file_path.name}: {e}")
            had_errors = True

    if processed == 0:
        print("No files matched the specified period.")
        sys.exit(1)

    if had_errors:
        print("\nCompleted with errors.")
        sys.exit(1)

    print("\nAll files processed successfully.")


if __name__ == "__main__":
    main()
