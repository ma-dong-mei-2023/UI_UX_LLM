import sys
import os
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent.parent

# Book source code package path
PKG_DIR = ROOT_DIR / "doc" / "source_code" / "LLMs-from-scratch" / "pkg"
SOURCE_DIR = ROOT_DIR / "doc" / "source_code" / "LLMs-from-scratch"

# Add pkg to Python path so we can import llms_from_scratch
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

# Data directories
BACKEND_DIR = Path(__file__).parent
DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "db.sqlite"

# Default training text
DEFAULT_TEXT_PATH = SOURCE_DIR / "ch02" / "01_main-chapter-code" / "the-verdict.txt"

# Instruction data
INSTRUCTION_DATA_PATH = SOURCE_DIR / "ch07" / "01_main-chapter-code" / "instruction-data.json"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# GPT-2 config presets
GPT_PRESETS = {
    "tiny": {
        "vocab_size": 50257,
        "context_length": 64,
        "emb_dim": 48,
        "n_heads": 2,
        "n_layers": 2,
        "drop_rate": 0.1,
        "qkv_bias": False,
    },
    "small": {
        "vocab_size": 50257,
        "context_length": 128,
        "emb_dim": 128,
        "n_heads": 4,
        "n_layers": 4,
        "drop_rate": 0.1,
        "qkv_bias": False,
    },
    "gpt2-124m": {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": True,
    },
    "gpt2-355m": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "drop_rate": 0.1,
        "qkv_bias": True,
    },
}
