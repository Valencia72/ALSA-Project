# Aspect-Level Sentiment Analysis (ALSA) for AI User Experience

## Project Overview

This project focuses on **Aspect-Level Sentiment Analysis (ALSA)** on AI-related user experience discussions collected from Reddit. The goal is to identify sentiment (**Positive, Negative, Neutral**) with respect to specific aspects such as performance, usability, trust, privacy, usefulness, and emotional impact.

Unlike traditional sentiment analysis, ALSA evaluates sentiment relative to a given aspect rather than assigning a single sentiment to the entire text.

The project uses:

- **TF-IDF + LinearSVC (Baseline Model)**
- **BERT-base (Transformer Model)**
- **DistilBERT (Final Selected Model)**
- **LIME (Explainability for SVM)**
- **Captum Integrated Gradients (Explainability for DistilBERT)**
- **Hugging Face Spaces + Gradio (Deployment)**

---

## Problem Statement

AI tools such as ChatGPT, Claude, Gemini, and Copilot generate mixed user experiences across multiple dimensions like speed, trust, privacy, cost, and usability. Traditional sentiment analysis fails to capture these fine-grained opinions.

This project solves that by performing **Aspect-Level Sentiment Analysis**, enabling sentiment prediction for each aspect individually.

Example:

**Text:**

> ChatGPT is fast but sometimes gives wrong answers.

**Aspect-wise Output:**

- performance → Positive
- trust_reliability → Negative

---

## Dataset Collection

Reddit posts were collected using the **PullPush Reddit API** from multiple technical and AI-related subreddits:

- learnmachinelearning
- deeplearning
- computerscience
- programming
- aiethics
- datascience
- machinelearning
- cscareerquestions
- learnprogramming
- codinghelp
- promptengineering

### Total Raw Collection

Approximately **200,000 posts** were collected.

---

## Data Preprocessing

### Steps Performed

### 1. Data Cleaning

- Removed null values
- Removed duplicates
- Standardized text fields
- Combined multiple cleaned datasets into one CSV

### 2. Strict UX Filtering

Only AI-related user experience posts were retained using:

- AI keywords
- UX keywords
- Drop keywords for irrelevant content

### 3. Aspect Filtering

Posts were filtered based on predefined ALSA aspects.

### 4. Manual ALSA Annotation

Each record was manually labeled for:

- aspect
- sentiment

Cross-verification was performed using LLM assistance and manual validation.

---

## Final ALSA Dataset

### Final Rows

**1426 manually labeled records**

### Sentiment Classes

- Positive
- Neutral
- Negative

### Aspect Categories

1. usefulness
2. performance
3. trust_reliability
4. creativity_productivity
5. frustration_limitations
6. emotional_impact
7. usability
8. societal_ethical
9. mental_health
10. efficiency
11. cost_value
12. privacy_security

### Sentiment Distribution

- Negative → 637
- Positive → 406
- Neutral → 383

---

## Methodology

## Step 1: Label Encoding

Sentiment labels were encoded as:

- Negative → 0
- Neutral → 1
- Positive → 2

---

## Step 2: Aspect-Aware Input Creation

Input format used:

```text
[ASPECT] aspect_name [TEXT] original_text
```

Example:

```text
[ASPECT] performance [TEXT] ChatGPT is fast but sometimes crashes
```

This allows the model to learn sentiment relative to a specific aspect.

---

## Step 3: Train/Test Split

- Train → 80%
- Test → 20%
- Stratified split used for label balance

---

## Models Used

Note: Due to GitHub file size limitations, the trained DistilBERT model files (.safetensors and tokenizer files) are not included in this repository. The deployed model is hosted separately on Hugging Face Spaces for real-time inference and demonstration.

# Model 1 — TF-IDF + LinearSVC

Used as the traditional machine learning baseline.

### Why Used

- Fast
- Strong baseline
- Easy explainability with LIME

---

# Model 2 — BERT-base

Used for contextual transformer-based classification.

### Why Used

- Deep contextual understanding
- Strong NLP benchmark

---

# Model 3 — DistilBERT (Final Model)

Selected as the final model.

### Why Used

- Faster than BERT
- Lower computational cost
- Strong performance
- Better deployment suitability

---

## Evaluation

### Metrics Used

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Per-Aspect Accuracy

### Example Per-Aspect Results

- frustration_limitations → 0.975
- performance → 0.909
- trust_reliability → 0.867
- usefulness → 0.852

---

## Explainable AI (XAI)

## LIME

Used for:

- TF-IDF + SVM explainability

Helps identify which words influenced prediction.

---

## Captum Integrated Gradients

Used for:

- DistilBERT explainability

Helps generate:

- token importance scores
- token-level heatmaps

This improves model transparency and interpretability.

---

## Deployment

The final DistilBERT model was deployed using:

- **Hugging Face Spaces**
- **Gradio Interface**

### Deployment Features

User provides:

- aspect
- text

System predicts:

- Positive / Negative / Neutral

### Live Demo

Deployed successfully for real-time sentiment prediction.

---

## Project Structure

```text
ALSA-Project/
│
├── config/
├── data/
├── models/
│   └── distilbert/
├── src/
│   ├── datacollection.py
│   ├── filtering.py
│   ├── prepare_alsa.py
│   ├── train_svm.py
│   ├── train_bert.py
│   ├── train_distilbert.py
│   ├── evaluation.py
│   ├── lime_explainability.py
│   └── xai_captum.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Transformers
- Hugging Face
- Datasets
- Captum
- LIME
- Gradio
- Matplotlib
- Seaborn

---

## Conclusion

This project successfully performs **Aspect-Level Sentiment Analysis** for AI user experience using transformer-based models.

The combination of:

- aspect-aware modeling
- DistilBERT fine-tuning
- explainable AI
- live deployment

makes the system practical, interpretable, and suitable for real-world sentiment analysis applications.

The project demonstrates strong academic contribution in:

- NLP
- Explainable AI
- Transformer-based sentiment analysis
- Human-AI interaction research

---
