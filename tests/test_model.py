from datasets import load_dataset
from pytest import fixture
from transformers.pipelines import pipeline

from src.config import MODELS_DIR


@fixture(scope="module")
def model():
    return pipeline("text-classification", str(MODELS_DIR / "distilbert-imdb"))


def test_model_accuracy(model):
    test_ds = load_dataset()
