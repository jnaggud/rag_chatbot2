"""
Simplified script to run the Streamlit interface.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_streamlit():
    """Run the Streamlit app with minimal settings to avoid conflicts."""
    # Set critical environment variables 
    os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    
    # Get path to the app
    streamlit_script = Path(__file__).parent / "streamlit_app.py"
    
    # Print instructions
    print("\n" + "="*80)
    print("Starting Streamlit server...")
    print("If the browser doesn't open automatically, manually go to: http://localhost:8501")
    print("="*80 + "\n")
    
    # Use subprocess instead of manipulating sys.argv
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(streamlit_script),
        "--server.port=8501"
    ]
    
    subprocess.run(cmd)
    sys.exit(0)  # Exit cleanly

if __name__ == "__main__":
    run_streamlit()
