import torchaudio

def load_and_prepare_audio(path: str, target_sr: int = 48000, min_seconds: float = 10.0):
    """Load audio, resample to target_sr, and pad to at least min_seconds."""
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    min_samples = int(target_sr * min_seconds)
    current = waveform.shape[-1]
    if current < min_samples:
        pad = min_samples - current
        waveform = torchaudio.functional.pad(waveform, (0, pad))

    return {"waveform": waveform, "sample_rate": sr}