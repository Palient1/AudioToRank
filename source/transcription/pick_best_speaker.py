from pyannote.core import Segment

def pick_best_speaker(segment_start: float, segment_end: float, diarization) -> str:
    best = "Unknown"
    best_overlap = 0.0

    target = Segment(segment_start, segment_end)

    # diarization is pyannote.core.Annotation
    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
        overlap = target & turn  # intersection Segment or empty
        if overlap:
            overlap_dur = overlap.duration
            if overlap_dur > best_overlap:
                best_overlap = overlap_dur
                best = speaker_label

    return best
