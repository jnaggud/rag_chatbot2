"""
Ollama model management for the RAG chatbot.
"""

import logging
import subprocess
import sys

logger = logging.getLogger(__name__)

def check_and_install_ollama():
    """Check if Ollama is installed and provide installation instructions if not."""
    try:
        result = subprocess.run(["ollama", "version"], capture_output=True, text=True)
        logger.info(f"Ollama is already installed: {result.stdout.strip()}")
    except FileNotFoundError:
        logger.info("Ollama is not installed. Please install Ollama manually:")
        logger.info("macOS: curl -fsSL https://ollama.com/install.sh | sh")
        logger.info("After installing, please restart this script.")
        sys.exit(1)

def pull_ollama_model(model_name="llama3"):
    """Pull the specified model in Ollama."""
    logger.info(f"Pulling {model_name} model from Ollama...")
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        logger.info(f"{model_name} model pulled successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error pulling {model_name} model: {e}")
        sys.exit(1)
