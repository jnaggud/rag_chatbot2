"""
Persistence utilities for the RAG chatbot.
"""

import json
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def save_index_descriptions(descriptions: Dict[str, str], file_path: str = "./index_descriptions.json"):
    """
    Save index descriptions to a JSON file.
    
    Args:
        descriptions: Dictionary of index name to description
        file_path: Path to save the JSON file
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(descriptions, f, indent=4)
        logger.info(f"Saved {len(descriptions)} index descriptions to {file_path}")
    except Exception as e:
        logger.error(f"Error saving index descriptions: {e}")

def load_index_descriptions(file_path: str = "./index_descriptions.json") -> Dict[str, str]:
    """
    Load index descriptions from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary of index name to description, or empty dict if file doesn't exist
    """
    if not os.path.exists(file_path):
        logger.info(f"No index descriptions file found at {file_path}")
        return {}
    
    try:
        with open(file_path, 'r') as f:
            descriptions = json.load(f)
        logger.info(f"Loaded {len(descriptions)} index descriptions from {file_path}")
        return descriptions
    except Exception as e:
        logger.error(f"Error loading index descriptions: {e}")
        return {}