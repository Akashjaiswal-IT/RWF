# RWF Monograph — Production-Deployable Full-Stack Application

Interactive research monograph and live Python computational workbench for **"Scalable Graph Classification via Random Walk Fingerprints"** (Peiyan Li, Honglian Wang, Christian Böhm — *IEEE ICDM 2024*, pp. 231–240).

---

## Option 1: Free Public HTTPS URL on Render (Recommended — 2 Minutes)
1. Unzip this folder and push it to a GitHub repository.
2. Go to [render.com](https://render.com) → **New +** → **Blueprint** → connect your repo.
3. Render automatically detects `render.yaml`, installs `requirements.txt`, runs `python start.py`, verifies `/healthz`, and gives you a public `https://rwf-monograph.onrender.com` link ready to share with your professor.

## Option 2: One-Click Deploy on Railway
1. Push this folder to GitHub.
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
3. Railway automatically reads `railway.json` & `Procfile`, binds to `$PORT`, and deploys immediately.

## Option 3: Docker (Any VPS / Cloud Run / Fly.io)
```bash
docker build -t rwf-monograph .
docker run -p 8000:8000 rwf-monograph
```
Open **http://localhost:8000** (and Swagger API docs at **http://localhost:8000/docs**).

## Option 4: Run Locally
```bash
pip install -r requirements.txt
python3 tests/test_backend.py   # Verify exact Part 4 math & all API routes
python3 start.py                # Serves http://localhost:8000
```

## Production Safety Built-In
- **Pydantic v1 & v2 compatible** (`_to_dict` shim works on all FastAPI/Pydantic versions).
- **Dynamic `$PORT` & `0.0.0.0` binding** via `start.py`.
- **Cloud-safe SQLite**: uses local `rwf_studio.db`, customizable via `RWF_DB_PATH`, with automatic `/tmp/rwf_studio.db` fallback on read-only containers.
- **Health checks**: `GET /healthz` and `GET /api/health` included for zero-downtime cloud probes.
