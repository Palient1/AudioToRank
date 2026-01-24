#!/bin/bash
set -e

# Start ollama in background
ollama serve &
OLLAMA_PID=$!

# Wait for ollama to start
sleep 5

# Pull gemma3 model
ollama pull gemma3

# Keep container running
wait $OLLAMA_PID
