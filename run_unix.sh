#!/usr/bin/env bash
# UDI - setup and start for macOS/Linux. Stops at the first failing step.
# Usage: ./run_unix.sh [--retrain-reference]
set -euo pipefail
cd "$(dirname "$0")"

command -v python3 >/dev/null || { echo "[ERROR] python3 not found"; exit 1; }
command -v npm >/dev/null || { echo "[ERROR] npm not found (install Node.js 18+)"; exit 1; }

cd backend
[ -x .venv/bin/python ] || python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ ! -f .env ]; then cp .env.example .env; echo "Created backend/.env from .env.example."; fi
if grep -q "YOUR_GEMINI_API_KEY" .env; then
  echo "[ACTION NEEDED] Edit backend/.env: set GEMINI_API_KEY and GEMINI_MODEL. UDI starts without them but cannot answer questions."
fi

if [ "${1:-}" = "--retrain-reference" ]; then
  python sample_data/generate_samples.py
  python train_models.py
fi

cd ../frontend
npm install
npm run build

cd ../backend
echo "Open http://localhost:5000"
python app.py
