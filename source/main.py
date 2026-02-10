import os
import sys
import warnings
import argparse
from datetime import datetime, time, timezone
from pathlib import Path

import torch
import whisper
import torch.nn.functional as F
import requests
from dotenv import load_dotenv

from pyannote.audio import Pipeline
from pyannote.audio.core.task import Problem, Resolution, Specifications

from prepare_audio import load_and_prepare_audio
from transcription.pick_best_speaker import pick_best_speaker

load_dotenv()

SAFE_GLOBALS = [
    Problem, 
    Specifications, 
    Resolution,
    torch.torch_version.TorchVersion
]

if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals(SAFE_GLOBALS)


HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

DEFAULT_INPUT_DIR = (
    Path("/app/audio_files")
    if Path("/app/audio_files").exists()
    else Path(__file__).resolve().parent.parent / "audios"
)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
PYANNOTE_MODEL = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")
AGENT_API = os.getenv("LLM_AGENT_HOST", "http://localhost:5001") + "/analyze"


warnings.filterwarnings("ignore")


def parse_user_datetime(value: str, is_end: bool) -> datetime:
    if len(value) == 10 and value[4] == "-" and value[7] == "-":
        day = datetime.strptime(value, "%Y-%m-%d").date()
        return datetime.combine(day, time.max if is_end else time.min)

    cleaned = value.replace("Z", "+00:00")
    if len(cleaned) >= 5 and (cleaned[-5] in ["+", "-"] and cleaned[-3] != ":"):
        cleaned = cleaned[:-2] + ":" + cleaned[-2:]
    return datetime.fromisoformat(cleaned)


def parse_filename(path: Path) -> tuple[str, datetime, str] | None:
    stem = path.stem
    parts = stem.split("%")
    if len(parts) < 4:
        return None

    full_name = " ".join(parts[:3])
    dt_str = parts[3]

    try:
        cleaned = dt_str.replace("Z", "+00:00")
        if len(cleaned) >= 5 and (cleaned[-5] in ["+", "-"] and cleaned[-3] != ":"):
            cleaned = cleaned[:-2] + ":" + cleaned[-2:]
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return None

    return full_name, dt, dt_str


def normalize_for_compare(value: datetime, reference: datetime) -> datetime:
    if value.tzinfo and reference.tzinfo:
        return value.astimezone(timezone.utc)
    if value.tzinfo and not reference.tzinfo:
        return value.replace(tzinfo=None)
    if not value.tzinfo and reference.tzinfo:
        return value
    return value


def should_process(file_dt: datetime, start_dt: datetime | None, end_dt: datetime | None) -> bool:
    if start_dt:
        if normalize_for_compare(file_dt, start_dt) < normalize_for_compare(start_dt, file_dt):
            return False
    if end_dt:
        if normalize_for_compare(file_dt, end_dt) > normalize_for_compare(end_dt, file_dt):
            return False
    return True


def analyze_file(
    file_path: Path,
    whisper_model,
    diarization_pipeline,
    device: torch.device,
    full_name: str,
    record_time: str,
):
    print(f"\nProcessing audio file: {file_path}\n")

    print("Step 1/3: Transcribing audio with Whisper...")
    transcript = whisper_model.transcribe(str(file_path))["segments"]
    print("Transcription complete.\n")

    print("Step 2/3: Running speaker diarization with pyannote...")
    audio_input = load_and_prepare_audio(str(file_path))

    diarization_pipeline.instantiate({
        "segmentation": {
            "min_duration_off": 0.1
        },
        "clustering": {
            "threshold": 0.9
        }
    })

    diarization = diarization_pipeline(
        audio_input,
        min_speakers=2,
        max_speakers=3
    )

    print("Diarization complete.\n")

    print("Step 3/3: Merging transcription with speaker labels...\n")

    formatted_segments = []
    for segment in transcript:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]

        speaker = pick_best_speaker(start, end, diarization)
        formatted_segments.append({"speaker": speaker, "text": text, "start": start, "end": end})

    print("\nProcessing finished!")

    print("\nStep 4/4: Generating analysis via LLM Agent...\n")

    full_text = "\n".join([f"[{segment['speaker']}]: {segment['text']}"
                           for segment in formatted_segments])

    print("Worker name: " + full_name)
    print("Record time: " + record_time)
    print("Dialog for analysis:")
    print(f"{full_text}\n")

    payload = {
        "dialog": full_text,
        "full_name": full_name,
        "record_time": record_time
    }

    print(f"Sending request to {AGENT_API} with timeout=600...")
    response = requests.post(AGENT_API, json=payload, timeout=600)
    if response.status_code == 200:
        result = response.json()
        status = result.get("status", "unknown")
        analysis = result.get("analysis", "No response")

        print(f"Analysis:\n{analysis}")
        print(f"\nStatus: {status}")

        if result.get("db_saved"):
            print("✓ Result saved to database")
        else:
            db_error = result.get("db_error", "Unknown error")
            print(f"✗ Failed to save to database: {db_error}")
        return True

    if response.status_code == 400:
        result = response.json()
        error = result.get("error", "Unknown error")
        raw_response = result.get("raw_response", "")
        print(f"Error parsing LLM response: {error}")
        if raw_response:
            print(f"Raw LLM response:\n{raw_response}")
        return False

    print(f"Error: {response.status_code}")
    try:
        result = response.json()
        error = result.get("error", response.text)
    except Exception:
        error = response.text
    print(f"Details: {error}")
    return False


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

    print("Loading pyannote diarization pipeline...")
    if torch.cuda.is_available():
        print("Using GPU for pyannote.")
        torch_device = "cuda"
    else:
        print("Using CPU for pyannote.")
        torch_device = "cpu"

    device = torch.device(torch_device)
    diarization_pipeline = Pipeline.from_pretrained(
        PYANNOTE_MODEL,
        use_auth_token=HUGGINGFACE_TOKEN,
    )
    diarization_pipeline.to(device)

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
            ok = analyze_file(file_path, whisper_model, diarization_pipeline, device, full_name, record_time)
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
