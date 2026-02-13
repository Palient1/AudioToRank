# AudioToRank
A pipeline for conversational audio analysis that combines speech-to-text, speaker diarization, and LLM-based qualitative evaluation to measure politeness and adherence to communication standards for an individual speaker.

Uses Whisper, NVIDIA NeMo and gemma3.


## ATTENTION

Filenames must have this format:

"{Surname}%{Name}%{MiddleName}%{y}-{m}-{d}T{h}:{m}:{s}{TimeZone}.mp3"

# Launch

## 1) Setup

Python 3.11 is required.

```bash
pip install -r requirements-сpu.txt -r requirements-main.txt -r requirements.txt
```
You can download `requirements-gpu.txt` instead of `requirements-сpu.txt` if you are going to compute with Nvidia GPU.
Attention: order matters, `requirements-main.txt` should be installed after `requirements-сpu.txt` or `requirements-gpu.txt` to avoid conflicts.

## 2) Start services (DB + Ollama + Agent)

### No GPU
```bash
docker pull postgres:16.9
docker pull palient/ollama:latest
docker pull palient/llm-agent:latest
docker compose -f docker-compose.yml up data-base ollama llm-agent
```

### Nvidia
```bash
docker pull postgres:16.9
docker pull palient/ollama:latest
docker pull palient/llm-agent:latest
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up data-base ollama llm-agent
```

## 3) Run main program on a folder

Before running the main program, make sure to set up the `.env` file with the appropriate values for your environment. Furthermore, ensure that the audio files you want to process are named according to the specified format and are located in the correct directory (e.g., `./audios`).

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

# .env reference

```
# Hugging Face Token
HUGGINGFACE_TOKEN={hf_token}

# LLM Agent settings
LLM_AGENT_PORT=5001
LLM_MODEL={llm_model}
OLLAMA_HOST=http://ollama:11434

# Whisper model
WHISPER_MODEL={whisper_model}

# Main app settings
MAIN_APP_PORT=8000
LLM_AGENT_HOST=http://localhost:5001

```