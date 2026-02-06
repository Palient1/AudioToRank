import os
import torch
import whisper
import torch.nn.functional as F
import warnings
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
AUDIO_FILE_NAME = os.getenv("AUDIO_FILE", "test_extra.mp3")

if os.path.exists("/app/audio_files"):
    # Running in Docker
    AUDIO_FILE = f"/app/audio_files/{AUDIO_FILE_NAME}"
else:
    # Running locally - use absolute path
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    AUDIO_FILE = os.path.join(PROJECT_ROOT, "audios", AUDIO_FILE_NAME)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
PYANNOTE_MODEL = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")
AGENT_API = os.getenv("LLM_AGENT_HOST", "http://localhost:5001") + "/analyze"


warnings.filterwarnings("ignore")


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


print(f"\nProcessing audio file: {AUDIO_FILE}\n")


print("Step 1/3: Transcribing audio with Whisper...")
transcript = whisper_model.transcribe(AUDIO_FILE)["segments"]
print("Transcription complete.\n")


print("Step 2/3: Running speaker diarization with pyannote...")
audio_input = load_and_prepare_audio(AUDIO_FILE)

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

worker_name = " ".join(AUDIO_FILE_NAME.split("%")[:3])
record_time = AUDIO_FILE_NAME.split(".")[0].split("%")[3]

print("Worker name: " + worker_name)
print("Record time: " + record_time)
print("Dialog for analysis:")
print(f"{full_text}\n")

payload = {
    "dialog": full_text,
    "full_name": worker_name,
    "record_time": record_time
}

try:
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
            
    elif response.status_code == 400:
        result = response.json()
        error = result.get("error", "Unknown error")
        raw_response = result.get("raw_response", "")
        print(f"Error parsing LLM response: {error}")
        if raw_response:
            print(f"Raw LLM response:\n{raw_response}")
    else:
        print(f"Error: {response.status_code}")
        result = response.json()
        error = result.get("error", response.text)
        print(f"Details: {error}")
except requests.exceptions.ConnectionError:
    print(f"Error: Cannot connect to LLM Agent at {AGENT_API}")
    print("Make sure the LLM Agent service is running.")
except Exception as e:
    print(f"Failed to get analysis: {e}")
