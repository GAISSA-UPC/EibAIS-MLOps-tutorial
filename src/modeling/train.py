from datasets import load_dataset
from evaluate import load
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.config import MODELS_DIR, PROCESSED_DATA_DIR


def tokenize(examples):
    outputs = tokenizer(examples["text"], truncation=True)
    return outputs


metric = load("accuracy")


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

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")

tokenized_ds = ds.map(tokenize, batched=True)

training_args = TrainingArguments(
    num_train_epochs=1,
    output_dir=MODELS_DIR / "distilbert-imdb-checkpoint",
    push_to_hub=False,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
)

data_collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    data_collator=data_collator,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model(MODELS_DIR / "distilbert-imdb")