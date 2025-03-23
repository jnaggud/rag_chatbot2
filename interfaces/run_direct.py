"""
Direct launcher for Streamlit app that bypasses the file watcher issues.
"""

import os
import subprocess

# Set necessary environment variables
os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# Run streamlit directly with only valid command line arguments
cmd = [
    "streamlit", "run", 
    "interfaces/streamlit_app.py",
    "--server.port=8501",
    "--server.fileWatcherType=none"
]

# Execute the command
subprocess.run(cmd)