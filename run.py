"""
ScholarLens Quick Launcher.
Starts the Streamlit web application.
"""

import sys
import subprocess
from pathlib import Path

def main():
    app_path = Path(__file__).resolve().parent / "app.py"
    print("=" * 60)
    print("Launching ScholarLens Research Paper Digest Copilot...")
    print(f"App Entrypoint: {app_path}")
    print("=" * 60)
    
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nScholarLens stopped.")

if __name__ == "__main__":
    main()
