from dataclasses import dataclass


@dataclass
class DiarSegment:
    """A single diarization segment parsed from RTTM."""
    start: float
    duration: float
    speaker: str

    @property
    def end(self) -> float:
        return self.start + self.duration


def parse_rttm(rttm_path: str) -> list[DiarSegment]:
    """Parse an RTTM file into a list of DiarSegment objects."""
    segments = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            segments.append(DiarSegment(start=start, duration=duration, speaker=speaker))
    return segments


def pick_best_speaker(segment_start: float, segment_end: float, diarization: list[DiarSegment]) -> str:
    """Find the speaker with the largest overlap for a given time range."""
    best = "Unknown"
    best_overlap = 0.0

    for seg in diarization:
        overlap_start = max(segment_start, seg.start)
        overlap_end = min(segment_end, seg.end)
        overlap_dur = overlap_end - overlap_start

        if overlap_dur > best_overlap:
            best_overlap = overlap_dur
            best = seg.speaker

    return best
