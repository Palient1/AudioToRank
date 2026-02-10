# AudioToRank
A pipeline for conversational audio analysis that combines speech-to-text, speaker diarization, and LLM-based qualitative evaluation to measure politeness and adherence to communication standards for an individual speaker.

Uses Whisper, NVIDIA NeMo and gemma3.


## ATTENTION!

Filenames must have this format: \\
"{Surname}%{Name}%{MiddleName}%{y}-{m}-{d}T{h}:{m}:{s}{TimeZone}.mp3"

# Launch

## 1) Start services (DB + Ollama + Agent)

### No GPU
```bash
docker compose -f docker-compose.yml up --build data-base ollama llm-agent
```

### Nvidia
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build data-base ollama llm-agent
```

## 2) Run main program on a folder

### Process all files in a folder
```bash
python source/main.py --input-dir ./audios
```

### Process files within a period
```bash
python source/main.py --input-dir ./audios --start 2026-02-01 --end 2026-02-10
```

### Period with exact time
```bash
python source/main.py --input-dir ./audios --start 2026-02-01T08:00:00+03:00 --end 2026-02-01T12:00:00+03:00
```

# Requirements

nvidia-container-toolkit (for GPU support)