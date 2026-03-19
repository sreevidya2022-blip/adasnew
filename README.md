# Mercedes-Benz ADAS Intelligence Platform

Full-stack ML-based ADAS development platform for Mercedes-Benz trucks.

## 8 Modules
1. Performance Dashboard — real-time metrics, heatmaps, fleet status
2. Annotation Tool — draw bounding boxes on truck sensor imagery
3. Model Monitor — A/B testing, confusion matrix, scenario performance
4. Incident Analyzer — log, cluster, review ADAS failures
5. Simulation Hub — animated EU road testing environment
6. Compliance Dashboard — ISO 26262, SOTIF, UN R156
7. Live Telemetry — real-time sensor feeds + ML inference overlay
8. Explainability — Grad-CAM, feature importance, decision trees

## Deploy to Railway (3 steps)

1. Push to GitHub
2. Railway → New Project → Deploy from GitHub → select this repo
3. Add `GROQ_API_KEY` in Variables tab

App runs on port 5000. Railway will detect the Dockerfile automatically.

## Stack
- Frontend: Single HTML file (no build step)
- Backend: Flask + SQLAlchemy + httpx
- Database: SQLite (auto) / PostgreSQL (set DATABASE_URL)
- AI: Groq API via httpx
