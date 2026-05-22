from pathlib import Path
import os
import sys
import traceback

import uvicorn

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / ".run-logs"
LOG_DIR.mkdir(exist_ok=True)

os.chdir(ROOT)

with (LOG_DIR / "backend.pythonw.log").open("a", encoding="utf-8") as log:
    sys.stdout = log
    sys.stderr = log
    print("backend python launcher started", flush=True)
    try:
        uvicorn.run("backend.main:app", host="127.0.0.1", port=8000)
    except Exception:
        traceback.print_exc()
        raise
