from evaluate import load
from datasets import load_dataset
import numpy as np
from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer
from loguru import logger

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

logger.info("Evaluating the model...")

metric = load("accuracy")


def tokenize(examples):
    outputs = tokenizer(examples["text"], truncation=True)
    return outputs


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR / "distilbert-imdb")
model = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR / "distilbert-imdb")
trainer = Trainer(model=model, processing_class=tokenizer, compute_metrics=compute_metrics)

test_ds = load_dataset(
    "parquet",
    data_files={
        "test": str(PROCESSED_DATA_DIR / "test.parquet"),
    },
)
test_ds = test_ds.map(tokenize, batched=True)

test_results = trainer.evaluate(test_ds["test"])
logger.info(f"Test results: {test_results}")
