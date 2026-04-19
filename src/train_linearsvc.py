from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(train_df["input_text"])
X_test = vectorizer.transform(test_df["input_text"])

# Train SVM
svm_model = LinearSVC()
svm_model.fit(X_train, train_df["label"])

# Evaluation
svm_preds = svm_model.predict(X_test)

print("=== TF-IDF + SVM Results ===")
print(classification_report(test_df["label"], svm_preds))
