# import joblib


# # Load trained model
# model = joblib.load("model/fake_news_model.pkl")

# # Load TF-IDF vectorizer
# vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# def predict_news(news):
#     # Convert text into TF-IDF
#     news_tfidf = vectorizer.transform([news])

#     # Prediction
#     prediction = model.predict(news_tfidf)[0]

#     # Probability
#     probability = model.predict_proba(news_tfidf)[0]

#     fake_probability = probability[0]
#     real_probability = probability[1]

#     if prediction == 0:
#         result = "FAKE NEWS"
#         confidence = fake_probability
#     else:
#         result = "REAL NEWS"
#         confidence = real_probability

#     return result, confidence


# # =========================
# # User Input
# # =========================

# print("====================================")
# print("      FAKE NEWS DETECTION SYSTEM")
# print("====================================")

# news = input("\nEnter news article/headline:\n")

# result, confidence = predict_news(news)

# print("\nPrediction:", result)
# print("Confidence:", round(confidence * 100, 2), "%")


import joblib

from preprocessing import clean_text


model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


def predict_news(news: str):
    cleaned = clean_text(news)
    if not cleaned:
        return "INVALID INPUT", 0.0

    news_tfidf = vectorizer.transform([cleaned])
    prediction = model.predict(news_tfidf)[0]
    probability = model.predict_proba(news_tfidf)[0]

    fake_probability = probability[0]
    real_probability = probability[1]

    if prediction == 0:
        return "FAKE NEWS", fake_probability
    return "REAL NEWS", real_probability


if __name__ == "__main__":
    print("====================================")
    print("      FAKE NEWS DETECTION SYSTEM")
    print("====================================")

    news = input("\nEnter news article/headline:\n")
    result, confidence = predict_news(news)

    print("\nPrediction:", result)
    print("Confidence:", round(confidence * 100, 2), "%")