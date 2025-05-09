import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

root = os.getenv("ROOT")
if root is None:
    logger.error("PROJ_ROOT environment variable not set. Please set it in your .env file.")
    raise ValueError(
        "ROOT environment variable is not set in .env file. Use dot-env-template file to create .env file."
    )

ROOT_DIR = Path(root)
logger.info(f"ROOT_DIR path is: {ROOT_DIR}")

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = ROOT_DIR / "models"

SEED = 2025
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    logger.error("HF_TOKEN environment variable not set. Please set it in your .env file.")
    raise ValueError(
        "HF_TOKEN environment variable is not set in .env file. Use dot-env-template file to create .env file."
    )

# If tqdm is installed, configure loguru with tqdm.write

# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
