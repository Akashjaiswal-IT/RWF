#!/usr/bin/env bash
set -e
echo "Starting RWF Studio Full-Stack Server at http://127.0.0.1:8000 ..."
python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
