#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "python3.11 not found. Attempting to install..."
  OS_NAME="$(uname -s)"
  if [[ "$OS_NAME" == "Darwin" ]]; then
    if command -v brew >/dev/null 2>&1; then
      brew install python@3.11
    else
      echo "Homebrew not found. Install Python 3.11 manually: https://www.python.org/downloads/"
      exit 1
    fi
  elif [[ "$OS_NAME" == "Linux" ]]; then
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update
      sudo apt-get install -y python3.11 python3.11-venv
    else
      echo "Unsupported Linux package manager. Install Python 3.11 manually."
      exit 1
    fi
  else
    echo "Unsupported OS. Install Python 3.11 manually."
    exit 1
  fi
fi

python3.11 -m venv .venv

"$ROOT_DIR/.venv/bin/python" -m pip install --upgrade pip
"$ROOT_DIR/.venv/bin/python" -m pip install -r requirements-main.txt -r requirements.txt

echo "Setup complete. Activate with: source .venv/bin/activate"
