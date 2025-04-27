# config/settings.py

import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

# 1) Load .env overrides
load_dotenv()

# 2) Load YAML config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, "r") as f:
    _cfg = yaml.safe_load(f)

# 3) Respect TOKENIZERS_PARALLELISM
tpar = os.getenv(
    "TOKENIZERS_PARALLELISM",
    str(_cfg["environment"]["tokenizers_parallelism"])
)
os.environ["TOKENIZERS_PARALLELISM"] = tpar

#
# Persistence
#
PERSIST_DIRECTORY            = _cfg["persist_directory"]
DESCRIPTIONS_FILE            = _cfg["descriptions_file"]
DEFAULT_PERSIST_DIRECTORY    = PERSIST_DIRECTORY
DEFAULT_DESCRIPTIONS_FILE    = DESCRIPTIONS_FILE

#
# Chunking
#
CHUNK_SIZE                   = 1028 #was 256
CHUNK_OVERLAP                = 256
BREAKPOINT_THRESHOLD_AMOUNT  = _cfg["chunking"]["breakpoint_threshold_amount"]
BREAKPOINT_THRESHOLD_TYPE    = _cfg["chunking"]["breakpoint_threshold_type"]

#
# Retrieval
#
COARSE_TOP_K                 = 20
RERANK_TOP_K                 = _cfg["retrieval"]["rerank_top_k"]

#
# Models
#
DEFAULT_EMBEDDING_MODEL      = _cfg["embedding_model"]
DEFAULT_LLM_MODEL            = _cfg["llm"]["default_model"]
AVAILABLE_LLM_MODELS         = _cfg["llm"]["available_models"]

#
# Logging
#
LOG_LEVEL                    = _cfg["logging"]["level"]
