#!/usr/bin/env python
"""
Minimal Streamlit launcher that avoids all problematic options.
"""

import os
import subprocess
import sys

# Set critical environment variables to avoid PyTorch conflicts
os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# Get the path to the streamlit app
streamlit_app_path = os.path.join("interfaces", "streamlit_app.py")

# Very minimal command - only use the most essential options
cmd = [
    sys.executable, "-m", "streamlit", "run", 
    streamlit_app_path,
    "--server.port=8501"
]

print("Running command:", " ".join(cmd))
subprocess.run(cmd)
