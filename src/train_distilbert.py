import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# CONFIG

DATA_PATH = "alsa_data_clean_encoded.csv"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./distilbert_results"

TEST_SIZE = 0.2
RANDOM_SEED = 42
MAX_LENGTH = 256
NUM_LABELS = 3
EPOCHS = 3
BATCH_SIZE = 16

# LOAD DATA

print(" Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Create aspect-aware input
df["input_text"] = "[ASPECT] " + df["aspect"] + " [TEXT] " + df["text"]

# Stratified split
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df["label"],
    random_state=RANDOM_SEED
)

print("Train size:", len(train_df))
print("Test size:", len(test_df))

# TOKENIZATION

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


def tokenize(batch):
    return tokenizer(
        batch["input_text"],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )


train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# MODEL

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

# TRAINING CONFIG

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    seed=RANDOM_SEED
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# TRAIN


print("\n Training DistilBERT...")
trainer.train()

# EVALUATION


print("\n Evaluating...")
predictions = trainer.predict(test_dataset)

preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

print("\nClassification Report:")
print(classification_report(labels, preds))

print("\n Training Complete.")
