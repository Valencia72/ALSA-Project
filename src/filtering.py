import pandas as pd
import re

# ---------- CONFIG ----------
INPUT_CSV = "final_ai_user_experience_only_final.csv"
OUTPUT_CSV = "ai_user_experience_strict.csv"

# AI tools / platforms
AI_KEYWORDS = [
    "chatgpt", "gpt", "openai", "claude", "bard", "gemini",
    "midjourney", "dall", "copilot", "llm", "ai tool", "ai assistant"
]

# Words that signal USER EXPERIENCE
UX_KEYWORDS = [
    "i use", "i used", "i feel", "i felt", "i think", "i believe",
    "my experience", "in my experience",
    "it helps", "it helped", "it sucks", "it works", "it doesn't work",
    "love", "hate", "frustrated", "annoying", "useful", "helpful",
    "trust", "privacy", "confusing", "easy", "hard",
    "better", "worse", "problem", "issue"
]

# Words that indicate NON-UX content (auto drop)
DROP_KEYWORDS = [
    "roadmap", "how to build", "dataset", "training", "model",
    "architecture", "paper", "research", "theory",
    "career", "job", "salary", "interview",
    "learning path", "certification", "course",
    "writing", "novel", "story", "poem"
]

# ---------- HELPERS ----------
def contains_any(text, keywords):
    return any(k in text for k in keywords)

def is_user_experience(text):
    text = text.lower()

    has_ai = contains_any(text, AI_KEYWORDS)
    has_ux = contains_any(text, UX_KEYWORDS)
    has_drop = contains_any(text, DROP_KEYWORDS)

    return has_ai and has_ux and not has_drop

# ---------- LOAD ----------
df = pd.read_csv(INPUT_CSV)

print("Total rows before filtering:", len(df))

# ---------- FILTER ----------
df["text_lower"] = df["text"].astype(str).str.lower()
df_filtered = df[df["text_lower"].apply(is_user_experience)]

df_filtered = df_filtered.drop(columns=["text_lower"])

print("Rows after strict UX filtering:", len(df_filtered))

# ---------- SAVE ----------
df_filtered.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f" Clean UX-only dataset saved to: {OUTPUT_CSV}")
