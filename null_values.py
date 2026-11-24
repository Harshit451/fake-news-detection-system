import pandas as pd

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

print("Missing values in Fake.csv:")
print(fake.isnull().sum())

print("\nMissing values in True.csv:")
print(true.isnull().sum())