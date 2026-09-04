"""One-time wiring patch: mount the agentic-commerce router.

Run once from the repo root:  python3 wiring_api_patch.py

Idempotent — running it twice changes nothing. It appends two lines to
backend/app/api.py and ensures the new tables are created by init_db().
"""
import pathlib
import sys

root = pathlib.Path(".")
api = root / "backend/app/api.py"
models = root / "backend/app/db/models.py"

if not api.exists():
    sys.exit("run this from the repo root (backend/app/api.py not found)")

s = api.read_text()
if "api_commerce" in s:
    print("api.py already wired — skipping")
else:
    s += (
        "\n\n# ------------------------------------------------- agentic commerce"
        "\n# Track 01. Additive: no existing route changes behaviour."
        "\nfrom backend.app.api_commerce import router as agent_commerce_router  # noqa: E402"
        "\n\napp.include_router(agent_commerce_router)\n"
    )
    api.write_text(s)
    print("api.py wired")

m = models.read_text()
if "commerce_models" in m:
    print("models.py already imports commerce_models — skipping")
else:
    m = m.replace(
        "def init_db(",
        "# Importing the negotiation tables here registers them on Base.metadata\n"
        "# so init_db() creates them with no migration step.\n"
        "def _register_commerce_models() -> None:\n"
        "    from backend.app.db import commerce_models  # noqa: F401\n"
        "\n\ndef init_db(",
        1,
    )
    # call it inside init_db
    lines = m.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("def init_db("):
            j = i + 1
            while j < len(lines) and (lines[j].strip().startswith('"""')
                                      or lines[j].strip() == ""):
                j += 1
            lines.insert(j, "    _register_commerce_models()")
            break
    models.write_text("\n".join(lines) + "\n")
    print("models.py wired")

print("\nNow run:  python3 -c \"from backend.app.db.models import init_db; init_db()\"")