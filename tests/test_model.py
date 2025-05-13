from datasets import load_dataset
import evaluate
import pytest
from transformers.pipelines import pipeline

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, SEED

MODEL_NAME = "distilbert-imdb"
ACC_THRESHOLD = 0.95


@pytest.fixture(scope="module")
def pipe():
    return pipeline("text-classification", str(MODELS_DIR / MODEL_NAME))


@pytest.fixture(scope="function")
def test_ds():
    ds = (
        load_dataset(
            "parquet",
            data_files={
                "test": str(PROCESSED_DATA_DIR / "test.parquet"),
            },
        )["test"]
        .shuffle(SEED)
        .select(range(1000))
    )
    return ds


def test_model_accuracy(pipe, test_ds):
    """
    Test the model's accuracy on a small test set.
    """
    accuracy = evaluate.load("accuracy")
    eval = evaluate.evaluator("text-classification")
    result = eval.compute(
        model_or_pipeline=pipe,
        data=test_ds,
        metric=accuracy,
        label_column="labels",
        label_mapping={"negative": 0, "positive": 1},
    )
    acc_score = result["accuracy"]
    assert acc_score > ACC_THRESHOLD, (
        f"Model accuracy is {acc_score:.2f}, which is below the threshold of {ACC_THRESHOLD}."
    )


@pytest.mark.parametrize(
    "text1, text2, label",
    [
        ("This movie is great!", "The movie isn't bad at all.", "positive"),
        ("The acting was terrible.", "I was expecting the acting to be better.", "negative"),
    ],
)
def test_negation(pipe, text1, text2, label):
    """
    Test the model's ability to handle negation in text.
    """
    result = pipe([text1, text2])

    assert result[0]["label"] == result[1]["label"], (
        "Model predicted different labels for negated sentences. "
        + f"It predicted {result[0]['label']} for '{text1}' but {result[1]['label']} for '{text2}'."
    )
    assert result[0]["label"] == label, f"Model predicted {result[0]['label']} for '{text1}', expected {label}"


@pytest.mark.parametrize(
    "text, expected_label",
    [
        ("This movie is great!", "positive"),
        ("This movie is terrible!", "negative"),
        ("I loved the plot and the characters.", "positive"),
        ("I hated the polot and the characters.", "negative"),
    ],
)
def test_model_predictions(pipe, text, expected_label):
    """
    Test the model's predictions when changing a single word in the text.
    """
    result = pipe(text)
    assert result[0]["label"] == expected_label, (
        f"Model predicted {result[0]['label']} for '{text}', expected {expected_label}."
    )
