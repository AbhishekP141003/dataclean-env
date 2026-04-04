"""
main.py — Starts FastAPI (port 7860) and Gradio UI (port 7861) together.
HF Spaces exposes port 7860 by default.
"""
import subprocess
import sys
import time
import threading
import uvicorn
from app import app


def run_gradio():
    time.sleep(3)  # Wait for FastAPI to be ready
    subprocess.Popen([sys.executable, "gradio_ui.py"])


if __name__ == "__main__":
    # Start Gradio in background thread
    t = threading.Thread(target=run_gradio, daemon=True)
    t.start()
    # Run FastAPI on primary port
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
