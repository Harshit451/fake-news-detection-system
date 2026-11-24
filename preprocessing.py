# import pandas as pd

# # Load datasets
# fake = pd.read_csv("dataset/Fake.csv")
# true = pd.read_csv("dataset/True.csv")

# # Add labels
# fake["label"] = 0
# true["label"] = 1

# # Combine datasets
# df = pd.concat([fake, true], ignore_index=True)

# # Remove duplicate records
# df = df.drop_duplicates()

# # Combine title and text
# df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

# # Shuffle the final dataset
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# # Display final pre-processed dataset
# print(df[["title", "text", "subject", "date", "label"]].head(10))

# # Check label distribution
# print("\nLabel Distribution:")
# print(df["label"].value_counts())



import re


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"^.*?\(Reuters\)\s*-\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_and_clean(title: str, body: str) -> str:
    title = title if isinstance(title, str) else ""
    body = body if isinstance(body, str) else ""
    return clean_text(f"{title} {body}")