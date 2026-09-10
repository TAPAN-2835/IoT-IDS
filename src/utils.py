import os
import json
import logging
from typing import Any, Dict

def setup_logger(name: str) -> logging.Logger:
    """Configure and return a standard logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary to a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def load_json(filepath: str) -> Dict[str, Any]:
    """Load dictionary from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
