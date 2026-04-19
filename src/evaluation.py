from sklearn.metrics import classification_report

predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

print(classification_report(labels, preds))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(labels, preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Aspect-Level Sentiment")
plt.show()

#Per-Aspect Performance
test_df = test_df.copy()
test_df["pred"] = preds

aspect_results = {}

for aspect in test_df["aspect"].unique():
    subset = test_df[test_df["aspect"] == aspect]
    if len(subset) > 10:  # avoid very small samples
        acc = (subset["label"] == subset["pred"]).mean()
        aspect_results[aspect] = round(acc, 3)

print("Per-Aspect Accuracy:")
for k, v in sorted(aspect_results.items(), key=lambda x: x[1], reverse=True):
    print(f"{k}: {v}")
