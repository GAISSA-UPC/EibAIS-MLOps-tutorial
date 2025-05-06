import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

root = os.getenv("ROOT")
if root is None:
    raise ValueError(
        "ROOT environment variable is not set in .env file. Use dot-env-template file to create .env file."
    )

ROOT_DIR = Path(root)

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

SEED = 2025
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
