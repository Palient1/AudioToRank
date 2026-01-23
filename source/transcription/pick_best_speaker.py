def pick_best_speaker(segment_start: float, segment_end: float, diarization) -> str:
    best = "Unknown"
    best_overlap = 0.0
    for turn, speaker_label in diarization.speaker_diarization:
        overlap_start = max(segment_start, turn.start)
        overlap_end = min(segment_end, turn.end)
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = speaker_label
    return best