# 18-Hour Practical Build Checklist (MVP, Ship-Ready)

You should not try to build the full research-grade system in 18 hours.
Build a demo-ready MVP with:

1. One working API endpoint.
2. Real image upload flow.
3. Four module score placeholders or lightweight heuristics.
4. Weighted score engine and decision labels.
5. Simple dashboard showing result, image, and flags.
6. Docker, README, and GitHub push.

---

## 0) MVP Scope Lock (15 min)

- [ ] Freeze scope: no model training, no Kubernetes, no full Celery farm.
- [ ] Keep architecture shape, but use lightweight module logic for demo.
- [ ] Define success: upload image -> get score + decision + explanation in UI.

---

## 1) Workspace Setup (45 min)

- [ ] Create backend folder (FastAPI).
- [ ] Create frontend folder (React or simple static page).
- [ ] Create shared sample data folder with 8-12 test images.
- [ ] Add `.env.example`.
- [ ] Add `requirements.txt` and `README.md`.

Commands:

```powershell
cd C:\Users\JHASHANK\Downloads\Hack_Nocturne_26
mkdir v1\backend,v1\frontend,v1\sample_data,v1\docs -Force
```

---

## 2) Backend Skeleton (2 hours)

- [ ] Build `POST /api/v1/verify` endpoint.
- [ ] Accept image and basic metadata.
- [ ] Save file temporarily.
- [ ] Run four module functions in sequence (or async if fast enough).
- [ ] Return JSON with:
  - `authenticity_score`
  - `decision`
  - `layer_scores`
  - `flags`
  - `processing_time_ms`

Decision bands:

- [ ] 85-100: AUTO_APPROVE
- [ ] 60-84: FAST_TRACK
- [ ] 35-59: SUSPICIOUS
- [ ] 0-34: REJECT

---

## 3) Lightweight Module Logic (3 hours)

Use fast heuristics now, but keep module names stable so you can swap real models later.

- [ ] CNN module stub: basic image quality/noise/compression consistency heuristic.
- [ ] ViT module stub: CLIP-like placeholder score (random seed + deterministic feature proxy).
- [ ] GAN module stub: FFT artifact heuristic.
- [ ] OCR module: EasyOCR extract text + expiry regex check (if OCR fails, neutral score).

- [ ] Standardize each module output:

```json
{ "score": "0-100", "flags": [], "details": {} }
```

---

## 4) Score Engine + Explainability Output (1.5 hours)

- [ ] Implement weighted score:
  - CNN 0.35
  - ViT 0.30
  - GAN 0.20
  - OCR 0.15
- [ ] Add a simple "why" list from top contributing flags.
- [ ] Save processed image copy path (heatmap can be placeholder overlay for MVP).

---

## 5) Frontend Dashboard (2.5 hours)

- [ ] Upload image form.
- [ ] Display score + decision badge.
- [ ] Display four layer scores as bars.
- [ ] Show returned flags.
- [ ] Show uploaded image and analysis image side-by-side.
- [ ] Add a clean single-page layout.

---

## 6) Persistence + Logs (1.5 hours)

- [ ] Use SQLite for speed (instead of PostgreSQL now).
- [ ] Save each request/response as an audit row.
- [ ] Add basic logs for timing and failures.
- [ ] Add `GET /api/v1/status/{id}` backed by the DB row.

---

## 7) Docker + Run Scripts (1.5 hours)

- [ ] Dockerfile for backend.
- [ ] Dockerfile for frontend.
- [ ] `docker-compose.yml` for both.
- [ ] One command to run all services.

---

## 8) Testing + Demo Data (2 hours)

- [ ] Run 10 API test calls using sample images.
- [ ] Verify all decision thresholds trigger correctly.
- [ ] Verify bad input handling (no image, wrong format).
- [ ] Capture three screenshots for final demo.

Quick sanity commands:

```powershell
# backend
cd C:\Users\JHASHANK\Downloads\Hack_Nocturne_26\v1\backend
python -m venv C:\Users\JHASHANK\Downloads\Hack_Nocturne_26\.venv
C:\Users\JHASHANK\Downloads\Hack_Nocturne_26\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

---

## 9) Final Packaging (1 hour)

- [ ] Update README with:
  - Problem
  - Architecture
  - How to run
  - API example request/response
  - Demo screenshots
- [ ] Add project structure section.
- [ ] Add "Known limitations" section (important for judging honesty).

---

## 10) GitHub Delivery Checklist (30 min)

- [ ] Confirm all files added.
- [ ] Commit with clear message.
- [ ] Push to `main`.
- [ ] Verify repo contains README + runnable instructions.

Commands:

```powershell
cd C:\Users\JHASHANK\Downloads\Hack_Nocturne_26
git add -A
git commit -m "Build VeriSight MVP: API, scoring engine, dashboard, docker, docs"
git push
```

If commit says "nothing to commit", run:

```powershell
git push
```

---

## Hard Cut List (Skip in 18 hours)

- Full model training/fine-tuning.
- Full SHAP/Grad-CAM pipeline.
- Celery distributed workers (optional).
- Kubernetes production deployment.
- Multi-cloud setup.
- Full benchmark experiments.

---

## Minimum Judge-Ready Demo Outcome

1. User uploads complaint image in UI.
2. API returns four layer scores plus a final authenticity score.
3. Decision is shown with threshold logic.
4. Flags and rationale are visible.
5. Audit entry is saved.
6. Project runs locally with a clear README in one command path.