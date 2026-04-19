import pandas as pd

# Reload fixed file
df = pd.read_csv("alsa_data_manual_final.csv")

# Normalize sentiment
df["sentiment"] = (
    df["sentiment"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Fix typos
df["sentiment"] = df["sentiment"].replace({
    "postive": "positive",
    "possitive": "positive"
})

# Keep only valid sentiments
valid = ["positive", "neutral", "negative"]
df = df[df["sentiment"].isin(valid)]

print(df["sentiment"].value_counts())
print("Rows:", len(df))

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["sentiment"])

print("Encoding map:")
print(dict(zip(label_encoder.classes_,
               label_encoder.transform(label_encoder.classes_))))

import pandas as pd

df = pd.read_csv("alsa_data_clean_encoded.csv")

df["input_text"] = (
    "[ASPECT] " + df["aspect"] + " [TEXT] " + df["text"]
)

df.head()

