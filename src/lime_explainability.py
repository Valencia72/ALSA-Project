from lime.lime_text import LimeTextExplainer
import numpy as np

class_names = ["negative", "neutral", "positive"]

def predict_proba(texts):
    X = vectorizer.transform(texts)
    decision = svm_model.decision_function(X)

    # Softmax conversion
    exp_scores = np.exp(decision)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return probs

explainer = LimeTextExplainer(class_names=class_names)

example_text = test_df["input_text"].iloc[0]

print("\n" + "-" * 60)
print("Input Text:")
print(example_text)
print("-" * 60)

exp = explainer.explain_instance(
    example_text,
    predict_proba,
    num_features=10
)

print("\nTop Important Features:")
for feature, weight in exp.as_list():
    print(f"{feature}: {round(weight, 4)}")
