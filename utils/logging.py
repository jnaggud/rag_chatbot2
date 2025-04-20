# utils/logging.py

import logging
from config.settings import LOG_LEVEL

def setup_logging():
    root = logging.getLogger()
    # Remove any existing handlers to avoid duplicates
    if root.handlers:
        root.handlers.clear()

    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(fmt)

    root.setLevel(LOG_LEVEL)
    root.addHandler(handler)
    return root
