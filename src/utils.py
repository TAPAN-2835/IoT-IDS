import os
import json
import logging
from pathlib import Path
from typing import Any, Dict

_LOG_FILE = Path(__file__).resolve().parent.parent / "pipeline.log"

def setup_logger(name: str) -> logging.Logger:
    """Configure and return a standard logger that streams to console AND
    appends to pipeline.log so the dashboard's /api/logs endpoint (and any
    other process) can tail live training progress off disk."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(_LOG_FILE, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.propagate = False
    return logger

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary to a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def load_json(filepath: str) -> Dict[str, Any]:
    """Load dictionary from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
