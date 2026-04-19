import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

from captum.attr import IntegratedGradients


# ---------------------------------------------------
# Load Saved DistilBERT Model
# ---------------------------------------------------

MODEL_PATH = "models/distilbert"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model.eval()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

print("Using device:", device)


# ---------------------------------------------------
# Sample Input Text
# ---------------------------------------------------

text = "asked chatgpt to make a patch based on what we talk about... i have issues"

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True
)

input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)


# ---------------------------------------------------
# Prediction Function for Captum
# ---------------------------------------------------

def predict(inputs_embeds, attention_mask):
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask
    )
    return outputs.logits


# ---------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------

embeddings = model.distilbert.embeddings.word_embeddings(
    input_ids
)

ig = IntegratedGradients(predict)

target_class = torch.argmax(
    model(input_ids, attention_mask).logits,
    dim=1
).item()

attributions = ig.attribute(
    embeddings,
    additional_forward_args=(attention_mask,),
    target=target_class,
    n_steps=50
)

attributions_sum = attributions.sum(dim=-1).squeeze(0)

tokens = tokenizer.convert_ids_to_tokens(
    input_ids.squeeze(0)
)


# ---------------------------------------------------
# Token-Level Importance Scores
# ---------------------------------------------------

token_attributions = list(
    zip(
        tokens,
        attributions_sum.detach().cpu().numpy()
    )
)

print("\nToken Importance Scores:\n")

for token, score in token_attributions:
    print(f"{token:15s} {score:.4f}")


# ---------------------------------------------------
# Normalize Token Importance Scores
# ---------------------------------------------------

scores = np.array([
    abs(score) for _, score in token_attributions
])

scores = scores / scores.max()

print("\nNormalized Token Scores:\n")

for token, score in zip(tokens, scores):
    print(f"{token:15s} {score:.3f}")


# ---------------------------------------------------
# Heatmap Preparation
# ---------------------------------------------------

filtered_tokens = []
filtered_scores = []

for token, score in zip(tokens, scores):
    if token not in ["[CLS]", "[SEP]", "[PAD]"]:
        filtered_tokens.append(token)
        filtered_scores.append(score)


# ---------------------------------------------------
# Heatmap Visualization
# ---------------------------------------------------

plt.figure(figsize=(len(filtered_tokens) * 0.6, 2))

sns.heatmap(
    [filtered_scores],
    annot=[filtered_tokens],
    fmt="",
    cmap="Reds",
    cbar=True
)

plt.title("Token-level Importance using Integrated Gradients")
plt.yticks([])

plt.savefig("captum_heatmap.png", bbox_inches="tight")
plt.show()

print("\nCaptum heatmap saved as: captum_heatmap.png")