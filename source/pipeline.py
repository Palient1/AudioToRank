from __future__ import annotations

import os
import tempfile
from pathlib import Path

import requests
from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf

from prepare_audio import convert_to_mono_wav
from transcription.pick_best_speaker import pick_best_speaker, parse_rttm
from diarization.nemo_config import load_nemo_diar_base_cfg


def analyze_file(
    file_path: Path,
    whisper_model,
    full_name: str,
    record_time: str,
    agent_api: str,
    diar_base_cfg,
    diar_yaml_path: Path,
) -> bool:
    print(f"\nProcessing audio file: {file_path}\n")

    print("Step 1/3: Transcribing audio with Whisper...")
    transcript = whisper_model.transcribe(str(file_path))["segments"]
    print("Transcription complete.\n")

    print("Step 2/3: Running speaker diarization with NeMo...")

    diar_out_root = Path(__file__).resolve().parent.parent / "output" / "diarization"
    try:
        diar_out_root.mkdir(parents=True, exist_ok=True)
        if not os.access(diar_out_root, os.W_OK):
            raise PermissionError(f"Not writable: {diar_out_root}")
        tmpdir = tempfile.mkdtemp(prefix="diar_", dir=str(diar_out_root))
    except PermissionError:
        tmpdir = tempfile.mkdtemp(prefix="diar_", dir=tempfile.gettempdir())
        print(f"[NeMo] Using fallback temp dir: {tmpdir}")
    print(f"[NeMo] Diarization output dir: {tmpdir}")

    try:
        wav_path = convert_to_mono_wav(str(file_path), tmpdir)

        # Create manifest for NeMo
        manifest_path = os.path.join(tmpdir, "manifest.json")
        import json

        manifest_entry = {
            "audio_filepath": wav_path,
            "offset": 0.0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": 2,
            "rttm_filepath": None,
            "uem_filepath": None,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest_entry, f)
            f.write("\n")

        # Configure NeMo diarizer
        cfg = OmegaConf.create(OmegaConf.to_container(diar_base_cfg, resolve=False))
        cfg.diarizer.manifest_filepath = manifest_path
        cfg.diarizer.out_dir = tmpdir
        print(f"[NeMo] Using diarization config: {diar_yaml_path}")

        diarizer = ClusteringDiarizer(cfg=cfg)
        diarizer.diarize()

        # Parse RTTM output
        rttm_dir = os.path.join(tmpdir, "pred_rttms")
        rttm_files = list(Path(rttm_dir).glob("*.rttm"))
        if not rttm_files:
            legacy_rttm_dir = os.path.join(tmpdir, "speaker_outputs", "pred_rttms")
            rttm_files = list(Path(legacy_rttm_dir).glob("*.rttm"))
            if rttm_files:
                rttm_dir = legacy_rttm_dir
        if not rttm_files:
            print("Warning: NeMo did not produce RTTM output")
            speaker_outputs = Path(tmpdir) / "speaker_outputs"
            if speaker_outputs.exists():
                print(f"[NeMo] speaker_outputs contents: {list(speaker_outputs.iterdir())}")
            else:
                print("[NeMo] speaker_outputs folder not found")
            diarization_segments = []
        else:
            diarization_segments = parse_rttm(str(rttm_files[0]))
    finally:
        pass

    print("Diarization complete.\n")

    print("Step 3/3: Merging transcription with speaker labels...\n")

    formatted_segments = []
    for segment in transcript:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]

        speaker = pick_best_speaker(start, end, diarization_segments)
        formatted_segments.append({"speaker": speaker, "text": text, "start": start, "end": end})

    print("\nProcessing finished!")

    print("\nStep 4/4: Generating analysis via LLM Agent...\n")

    full_text = "\n".join([f"[{segment['speaker']}]: {segment['text']}" for segment in formatted_segments])

    print("Worker name: " + full_name)
    print("Record time: " + record_time)
    print("Dialog for analysis:")
    print(f"{full_text}\n")

    payload = {
        "dialog": full_text,
        "full_name": full_name,
        "record_time": record_time,
    }

    print(f"Sending request to {agent_api} with timeout=600...")
    response = requests.post(agent_api, json=payload, timeout=600)
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
