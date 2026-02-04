# AudioToRank
A pipeline for conversational audio analysis that combines speech-to-text, speaker diarization, and LLM-based qualitative evaluation to measure politeness and adherence to communication standards for an individual speaker.

Uses Whisper, pyannote and gemma3.


# Launch

### No GPU
```
docker compose -f docker-compose.yml up --build
```

### Nvidia
```
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

# Requirements

nvidia-container-toolkit (for GPU support)