#!/usr/bin/env bash
# UDI 2.0 - setup and start for macOS/Linux. Stops at the first failing step.
# Usage: ./run_unix.sh [--retrain-reference]
set -euo pipefail
cd "$(dirname "$0")"

command -v python3 >/dev/null || { echo "[ERROR] python3 (3.11+) not found"; exit 1; }
command -v npm >/dev/null || { echo "[ERROR] npm not found (install Node.js 18+)"; exit 1; }

echo "=== 1/5 Python environment and packages ==="
cd backend
[ -x .venv/bin/python ] || python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "=== 2/5 Environment file ==="
if [ ! -f .env ]; then cp .env.example .env; echo "Created backend/.env from .env.example."; fi
if ! grep -qE "^GEMINI_API_KEY=.+" .env; then
  echo "[INFO] GEMINI_API_KEY is empty: UDI will plan questions with its local intent model (add a key to backend/.env to use Gemini)."
fi

echo "=== 3/5 Local planner model ==="
python train_planner.py

if [ "${1:-}" = "--retrain-reference" ]; then
  echo "=== optional: reference models ==="
  python sample_data/generate_samples.py
  python train_models.py
fi

echo "=== 4/5 Frontend build ==="
cd ../frontend
npm install
npm run build

echo "=== 5/5 Starting UDI on http://localhost:5000 (Ctrl+C to stop) ==="
cd ../backend
python serve.py
