# config/settings.py

import os
from pathlib import Path

from dotenv import load_dotenv
import yaml

# Load environment overrides first
load_dotenv()

# Load YAML config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, "r") as f:
    _cfg = yaml.safe_load(f)

# Enforce tokenizers parallelism setting
tpar = os.getenv("TOKENIZERS_PARALLELISM",
                 str(_cfg["environment"]["tokenizers_parallelism"]))
os.environ["TOKENIZERS_PARALLELISM"] = tpar

# Persistence
PERSIST_DIRECTORY = _cfg["persist_directory"]
DESCRIPTIONS_FILE = _cfg["descriptions_file"]

# Chunking
CHUNK_SIZE = _cfg["chunking"]["chunk_size"]
CHUNK_OVERLAP = _cfg["chunking"]["chunk_overlap"]
BREAKPOINT_THRESHOLD_AMOUNT = _cfg["chunking"]["breakpoint_threshold_amount"]
BREAKPOINT_THRESHOLD_TYPE = _cfg["chunking"]["breakpoint_threshold_type"]

# Retrieval
COARSE_TOP_K = _cfg["retrieval"]["coarse_top_k"]
RERANK_TOP_K = _cfg["retrieval"]["rerank_top_k"]

# Models
DEFAULT_EMBEDDING_MODEL = _cfg["embedding_model"]
DEFAULT_LLM_MODEL = _cfg["llm"]["default_model"]
AVAILABLE_LLM_MODELS = _cfg["llm"]["available_models"]

# Logging
LOG_LEVEL = _cfg["logging"]["level"]

# Default paths
DEFAULT_PERSIST_DIRECTORY   = "./chroma_db"
DEFAULT_DESCRIPTIONS_FILE   = "./index_descriptions.json"

# for backwards‑compatibility / easy imports:
DEFAULT_PERSIST_DIRECTORY   = PERSIST_DIRECTORY
DEFAULT_DESCRIPTIONS_FILE   = DESCRIPTIONS_FILE

