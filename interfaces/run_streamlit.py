"""
Script to run the Streamlit interface.
"""

import os
import streamlit.web.cli as stcli
import sys
from pathlib import Path

def run_streamlit():
    """Run the Streamlit app."""
    streamlit_script = Path(__file__).parent / "streamlit_app.py"
    
    # Set environment variables to control Streamlit behavior
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "false"  # Ensure browser opening
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    
    # Print clear instructions
    print("\n" + "="*80)
    print("Starting Streamlit server...")
    print("If the browser doesn't open automatically, manually go to: http://localhost:8501")
    print("="*80 + "\n")
    
    # Run the Streamlit app
    sys.argv = [
        "streamlit", "run", 
        str(streamlit_script),
        "--server.port=8501", 
        "--server.address=0.0.0.0",
        "--server.headless=false",  # This should open the browser
        "--browser.serverAddress=localhost",
        "--browser.gatherUsageStats=false"
    ]
    sys.exit(stcli.main())

if __name__ == "__main__":
    run_streamlit()