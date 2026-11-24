import pandas as pd

# Load original datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Build the exact 'content' column used in train.py
fake_clean = pd.DataFrame({
    "content": (fake["title"].fillna("") + " " + fake["text"].fillna("")).str.strip()
})

true_clean = pd.DataFrame({
    "content": (true["title"].fillna("") + " " + true["text"].fillna("")).str.strip()
})

# Remove empty rows
fake_clean = fake_clean[fake_clean["content"] != ""]
true_clean = true_clean[true_clean["content"] != ""]

# Save to two new CSV files
fake_clean.to_csv("dataset/fake_only_content.csv", index=False)
true_clean.to_csv("dataset/true_only_content.csv", index=False)

print("Saved dataset/fake_only_content.csv:", len(fake_clean), "rows")
print("Saved dataset/true_only_content.csv:", len(true_clean), "rows")