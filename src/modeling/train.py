from datasets import load_dataset
from evaluate import load
import mlflow
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
)

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, SEED

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
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

train_ds = ds["train"].shuffle(SEED).select(range(1000)).map(tokenize, batched=True)  # Select 1000 samples for training
train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
validation_ds = ds["validation"].shuffle(SEED).select(range(1000)).map(tokenize, batched=True)
validation_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

training_args = TrainingArguments(
    num_train_epochs=1,
    output_dir=MODELS_DIR / "distilbert-imdb-checkpoint",
    push_to_hub=False,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
)

data_collator = DataCollatorWithPadding(tokenizer)

id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, label2id=label2id, id2label=id2label
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

with mlflow.start_run() as run:
    trainer.train()

    finetuned_pipeline = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
    )

    mlflow.transformers.log_model(
        transformers_model=finetuned_pipeline,
        artifact_path="model",
        registered_model_name="distilbert-imdb",
        input_example=["This film seemed way too long even at only 75 minutes.", "I loved this movie!"],
    )

trainer.save_model(MODELS_DIR / "distilbert-imdb")
