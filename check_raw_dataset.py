import pandas as pd

fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")

print("Fake:", len(fake_df))
print("Real:", len(true_df))
print("Total:", len(fake_df) + len(true_df))