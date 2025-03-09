"""
Dependency management for the RAG chatbot.
"""

import logging
import sys
import subprocess

logger = logging.getLogger(__name__)

def check_and_install_dependencies():
    """Check and install all required dependencies for the RAG chatbot system."""
    required_packages = [
        "langchain", "langchain-community", "langchain-core", "langchain-text-splitters",
        "langchain-chroma", "chromadb", "sentence-transformers", "pydantic", "numpy",
        "fastapi", "uvicorn", "python-dotenv", "typer", "rich", "httpx", "pypdf"
    ]
    
    logger.info("Checking and installing dependencies...")
    
    try:
        import pip
        for package in required_packages:
            try:
                __import__(package.split('-')[0])
                logger.info(f"{package} is already installed.")
            except ImportError:
                logger.info(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"{package} has been installed.")
    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        sys.exit(1)
    
    logger.info("All dependencies are installed successfully.")
