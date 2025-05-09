from datasets import load_dataset
from evaluate import load
from loguru import logger
import mlflow
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, SEED

SPEEDUP_TRAINING = False  # Set to True to speed up training by using a smaller dataset

BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 2e-5
HF_MODEL_NAME = "distilbert/distilbert-base-uncased"
MODEL_NAME = "distilbert-imdb"

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
metric = load("accuracy")


def tokenize(examples):
    outputs = tokenizer(examples["text"], truncation=True)
    return outputs


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


ds = load_dataset(
    "parquet",
    data_files={
        "train": str(PROCESSED_DATA_DIR / "train.parquet"),
        "validation": str(PROCESSED_DATA_DIR / "validation.parquet"),
    },
)

if SPEEDUP_TRAINING:
    logger.info("Speeding up training by using a smaller dataset.")
    # Select 1000 samples for training
    train_ds = ds["train"].shuffle(SEED).select(range(1000)).map(tokenize, batched=True)
    validation_ds = ds["validation"].shuffle(SEED).select(range(1000)).map(tokenize, batched=True)
else:
    train_ds = ds["train"].map(tokenize, batched=True)
    validation_ds = ds["validation"].map(tokenize, batched=True)

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
validation_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

training_args = TrainingArguments(
    output_dir=MODELS_DIR / f"{MODEL_NAME}-checkpoint",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    push_to_hub=False,
)

data_collator = DataCollatorWithPadding(tokenizer)

id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained(
    HF_MODEL_NAME, num_labels=2, label2id=label2id, id2label=id2label, problem_type="single_label_classification"
).to(device)

trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    data_collator=data_collator,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=train_ds,
    compute_metrics=compute_metrics,
)

mlflow.set_experiment("IMDB sentiment analysis")
mlflow.set_system_metrics_sampling_interval(5)
with mlflow.start_run(log_system_metrics=True) as run:
    trainer.train()

trainer.save_model(MODELS_DIR / MODEL_NAME)

with mlflow.start_run(run_id=run.info.run_id, log_system_metrics=False):
    model_info = mlflow.transformers.log_model(
        transformers_model=str(MODELS_DIR / MODEL_NAME),
        task="text-classification",
        artifact_path="model",
        registered_model_name=MODEL_NAME,
        input_example=["This film seemed way too long even at only 75 minutes.", "I loved this movie!"],
    )

client = mlflow.MlflowClient()
client.set_registered_model_alias(MODEL_NAME, "challenger", model_info.registered_model_version)
