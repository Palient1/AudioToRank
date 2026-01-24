#!/bin/bash
set -e

# Wait for llm-agent to be ready
echo "Waiting for LLM Agent server to start..."
for i in {1..60}; do
    if curl -s http://llm-agent:5001/health > /dev/null 2>&1; then
        echo "LLM Agent server is running!"
        break
    fi
    echo "Attempt $i/60: LLM Agent not ready, waiting..."
    sleep 2
done


# Run main script
echo "Starting audio analysis..."
cd /app/source
python main.py

while true; do sleep 60; done