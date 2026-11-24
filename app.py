# import streamlit as st
# import joblib


# # =========================
# # Load Model
# # =========================

# model = joblib.load("model/fake_news_model.pkl")
# vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# # =========================
# # Page Configuration
# # =========================

# st.set_page_config(
#     page_title="Fake News Detection",
#     page_icon="📰",
#     layout="centered"
# )


# # =========================
# # Title
# # =========================

# st.title("📰 Fake News Detection System")

# st.write(
#     "Enter a news article or headline and the machine learning model "
#     "will classify it as Fake or Real."
# )


# # =========================
# # Input
# # =========================

# news = st.text_area(
#     "Enter News Content",
#     height=250,
#     placeholder="Paste the news article here..."
# )


# # =========================
# # Prediction
# # =========================

# if st.button("🔍 Detect News"):

#     if news.strip() == "":
#         st.warning("Please enter some news content.")

#     else:

#         # TF-IDF transformation
#         news_tfidf = vectorizer.transform([news])

#         # Prediction
#         prediction = model.predict(news_tfidf)[0]

#         # Probability
#         probabilities = model.predict_proba(news_tfidf)[0]

#         fake_probability = probabilities[0]
#         real_probability = probabilities[1]


#         # =========================
#         # Result
#         # =========================

#         if prediction == 0:

#             st.error("🚨 FAKE NEWS")

#             st.write(
#                 f"Fake probability: "
#                 f"{fake_probability * 100:.2f}%"
#             )

#             st.write(
#                 f"Real probability: "
#                 f"{real_probability * 100:.2f}%"
#             )

#         else:

#             st.success("✅ REAL NEWS")

#             st.write(
#                 f"Real probability: "
#                 f"{real_probability * 100:.2f}%"
#             )

#             st.write(
#                 f"Fake probability: "
#                 f"{fake_probability * 100:.2f}%"
#             )


# # =========================
# # Footer
# # =========================

# st.markdown("---")

# st.caption(
#     "Fake News Detection | Machine Learning Project | "
#     "TF-IDF + Logistic Regression"
# )



import joblib
import streamlit as st

from preprocessing import clean_text


st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("model/fake_news_model.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_artifacts()

st.title("📰 Fake News Detection System")
st.write(
    "Paste a news headline and full article text below. "
    "The model analyzes writing and vocabulary patterns using TF-IDF + Logistic Regression."
)

news = st.text_area(
    "Enter News Content (Headline + Article Body)",
    height=250,
    placeholder="Paste the news headline and full article text here...",
)

if st.button("🔍 Detect News"):
    cleaned_news = clean_text(news)
    word_count = len(cleaned_news.split())

    if word_count == 0:
        st.warning("Please enter some news content.")
    else:
        if word_count < 15:
            st.info(
                "Short text detected (under 15 words). Predictions are much more accurate "
                "when you include both the headline and full article body."
            )

        news_tfidf = vectorizer.transform([cleaned_news])
        prediction = model.predict(news_tfidf)[0]
        probabilities = model.predict_proba(news_tfidf)[0]

        fake_probability = probabilities[0]
        real_probability = probabilities[1]
        max_confidence = max(fake_probability, real_probability)

        if max_confidence < 0.65:
            st.warning(
                f"⚠️ UNCERTAIN RESULT (Real: {real_probability * 100:.2f}% | "
                f"Fake: {fake_probability * 100:.2f}%). The wording is close to the decision boundary."
            )
        elif prediction == 0:
            st.error("🚨 FAKE NEWS")
            st.write(f"Fake probability: **{fake_probability * 100:.2f}%**")
            st.write(f"Real probability: **{real_probability * 100:.2f}%**")
        else:
            st.success("✅ REAL NEWS")
            st.write(f"Real probability: **{real_probability * 100:.2f}%**")
            st.write(f"Fake probability: **{fake_probability * 100:.2f}%**")

st.markdown("---")
st.caption(
    "Fake News Detection | TF-IDF (1-2 n-grams) + Logistic Regression. "
)