"""RevenueOS API.

Phase 2 status: structural placeholder. The health endpoint and dataset summary
work today so deployment plumbing can be validated early; recovery, policy and
Razorpay routers land in Phases 4-6 under `backend/app/api/`.

Run: `make api`
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException

DATA = Path("data/generated")

app = FastAPI(
    title="RevenueOS",
    description="Autonomous Revenue Recovery for Intelligent Commerce",
    version="0.2.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "phase": "2", "dataset_present": (DATA / "manifest.json").exists()}


@app.get("/api/dataset/manifest")
def manifest() -> dict:
    """Seed, versions and record counts for the generated dataset."""
    path = DATA / "manifest.json"
    if not path.exists():
        raise HTTPException(404, "No dataset generated. Run `make data` first.")
    return json.loads(path.read_text())
