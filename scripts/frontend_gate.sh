#!/usr/bin/env bash
# Frontend gate: typecheck, build, and report. Run from the repository root.
set -uo pipefail
cd "$(dirname "$0")/../frontend" || exit 1

echo "==> typecheck"; npx tsc --noEmit; TS=$?
echo "==> build";     npm run build >/tmp/fe_build.log 2>&1; BUILD=$?
tail -20 /tmp/fe_build.log

cd ..
python3 - "$TS" "$BUILD" <<'PY'
import sys, subprocess
from datetime import datetime, timezone
from pathlib import Path

ts, build = int(sys.argv[1]), int(sys.argv[2])
routes = ["/", "/opportunities", "/opportunities/[id]", "/evaluation",
          "/agent", "/audit", "/settings"]
report = f"""# RevenueOS — Frontend Gate Report

Generated {datetime.now(timezone.utc).isoformat()}

## 1. Routes built

{chr(10).join(f'- `{r}`' for r in routes)}

## 2. Backend endpoints integrated

| endpoint | used by |
|---|---|
| `GET /health` | shell health indicator (15s poll) |
| `GET /api/version` | settings |
| `GET /api/opportunities` | queue, overview, audit |
| `GET /api/opportunities/{{id}}` | detail, audit timeline |
| `POST /api/opportunities/{{id}}/analyze` | detail |
| `POST /api/opportunities/{{id}}/approve` \\| `/reject` | approval gate |
| `POST /api/opportunities/{{id}}/execute` | execution panel + Razorpay checkout |
| `GET /api/dashboard/metrics` | overview (live operational) |
| `GET /api/evaluation/summary` | evaluation (research) |
| `GET /api/agent/summary` | agent safety |
| `GET /api/policy` | settings |

## 3. Opportunity detail features

- header with revenue at risk, state, attempt, Test Mode badge
- workflow stepper including the failure branch
- selected action with ΔEV, probability, cost and max downside
- conversion-versus-economics side-by-side comparison
- candidate table with table/chart toggle and explicit zero line
- why-this-action and why-not-alternatives
- policy panel with per-rule input, threshold and reason
- approval gate with reasoning before the buttons
- execution confirmation, Razorpay checkout, retry comparison
- categorised audit timeline with detail drawer

## 4. Design decisions worth noting

- Probability is never coloured; only economic value carries semantic colour,
  so a high-converting negative-ΔEV action cannot read as good.
- Research and live metrics are labelled separately and never combined.
- Polling is limited to states an external party can change.
- Payment recovery is shown as confirmed only after webhook verification.

## 5. Results

| check | result |
|---|---|
| TypeScript typecheck | {'PASS' if ts == 0 else 'FAIL'} |
| Production build | {'PASS' if build == 0 else 'FAIL'} |

## 6. Known limitations

- No Playwright end-to-end suite yet; verification is typecheck plus build plus
  manual route checks against a live backend.
- Charts are hand-built SVG/CSS rather than a charting library, which keeps the
  bundle small but limits interactivity.
- Global search covers opportunity IDs only.
- Demo mode is documented in `docs/demo-flow.md` rather than implemented as a
  guided in-app walkthrough.

## 7. Recommendation

**{'PASS' if ts == 0 and build == 0 else 'REVIEW REQUIRED'}**
"""
Path("evaluation/results").mkdir(parents=True, exist_ok=True)
Path("evaluation/results/frontend_gate_report.md").write_text(report)
print(report)
print("\nWrote evaluation/results/frontend_gate_report.md")
sys.exit(0 if ts == 0 and build == 0 else 1)
PY