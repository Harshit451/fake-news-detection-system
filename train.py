# import pandas as pd
# import os
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# # =========================
# # 1. Load Dataset
# # =========================

# fake_path = "dataset/Fake.csv"
# true_path = "dataset/True.csv"

# fake_news = pd.read_csv(fake_path)
# true_news = pd.read_csv(true_path)

# print("Fake news records:", len(fake_news))
# print("Real news records:", len(true_news))


# # =========================
# # 2. Add Labels
# # =========================

# fake_news["label"] = 0
# true_news["label"] = 1


# # =========================
# # 3. Combine Dataset
# # =========================

# data = pd.concat([fake_news, true_news], ignore_index=True)

# print("\nTotal records:", len(data))


# # =========================
# # 4. Create Text Column
# # =========================

# data["title"] = data["title"].fillna("")
# data["text"] = data["text"].fillna("")

# data["content"] = data["title"] + " " + data["text"]


# # =========================
# # 5. Remove Unnecessary Columns
# # =========================

# data = data[["content", "label"]]


# # =========================
# # 6. Shuffle Dataset
# # =========================

# data = data.sample(frac=1, random_state=42).reset_index(drop=True)


# # =========================
# # 7. Input and Output
# # =========================

# X = data["content"]
# y = data["label"]


# # =========================
# # 8. Train/Test Split
# # =========================

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )


# # =========================
# # 9. TF-IDF
# # =========================

# vectorizer = TfidfVectorizer(
#     stop_words="english",
#     max_df=0.7
# )

# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)

# print("\nTF-IDF transformation completed.")


# # =========================
# # 10. Train Model
# # =========================

# model = LogisticRegression(max_iter=1000)

# model.fit(X_train_tfidf, y_train)

# print("Model training completed.")


# # =========================
# # 11. Prediction
# # =========================

# y_pred = model.predict(X_test_tfidf)


# # =========================
# # 12. Evaluation
# # =========================

# accuracy = accuracy_score(y_test, y_pred)

# print("\n==============================")
# print("MODEL PERFORMANCE")
# print("==============================")

# print("Accuracy:", accuracy)

# print("\nClassification Report:")
# print(classification_report(
#     y_test,
#     y_pred,
#     target_names=["Fake", "Real"]
# ))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))


# # =========================
# # 13. Save Model
# # =========================

# os.makedirs("model", exist_ok=True)

# joblib.dump(model, "model/fake_news_model.pkl")
# joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

# print("\nModel saved successfully!")

# print("model/fake_news_model.pkl")
# print("model/tfidf_vectorizer.pkl")


import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from preprocessing import combine_and_clean


fake_news = pd.read_csv("dataset/Fake.csv")
true_news = pd.read_csv("dataset/True.csv")

fake_news["label"] = 0
true_news["label"] = 1

data = pd.concat([fake_news, true_news], ignore_index=True)

data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")

data = data.drop_duplicates(subset=["title", "text"]).reset_index(drop=True)

data["content"] = [
    combine_and_clean(t, b) for t, b in zip(data["title"], data["text"])
]

data = data[data["content"].str.split().str.len() >= 10].reset_index(drop=True)

print("Total clean records:", len(data))
print(data["label"].value_counts())

X = data["content"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    min_df=3,
    ngram_range=(1, 2),
    max_features=20000,
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, C=2.0)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\nUpdated model and vectorizer saved successfully!")