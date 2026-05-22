from pathlib import Path
import os
import sys
import traceback

from streamlit.web import cli as stcli

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / ".run-logs"
LOG_DIR.mkdir(exist_ok=True)

os.chdir(ROOT)
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

with (LOG_DIR / "frontend.pythonw.log").open("a", encoding="utf-8") as log:
    sys.stdout = log
    sys.stderr = log
    print("frontend python launcher started", flush=True)
    try:
        sys.argv = [
            "streamlit",
            "run",
            "frontend/app.py",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "8501",
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ]
        stcli.main()
    except Exception:
        traceback.print_exc()
        raise
