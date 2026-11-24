import pandas as pd
from preprocessing import combine_and_clean

fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df], ignore_index=True)
df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

df["content"] = [
    combine_and_clean(title, text)
    for title, text in zip(df["title"], df["text"])
]

df = df[df["content"].str.split().str.len() >= 10]
df = df.drop_duplicates(subset=["content"]).reset_index(drop=True)

print("Final total:", len(df))
print("Real (1):", (df["label"] == 1).sum())
print("Fake (0):", (df["label"] == 0).sum())