import os
import subprocess


def convert_to_mono_wav(input_path: str, output_dir: str, target_sr: int = 16000) -> str:
    """Convert any audio file to mono WAV at target_sr using ffmpeg.
    
    NeMo requires mono 16kHz WAV files for diarization.
    Returns the path to the converted WAV file.
    """
    basename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{basename}.wav")

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1",
        "-ar", str(target_sr),
        "-acodec", "pcm_s16le",
        output_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    return output_path