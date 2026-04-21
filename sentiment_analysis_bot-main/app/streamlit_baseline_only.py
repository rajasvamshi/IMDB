import joblib
import streamlit as st

# Load baseline models
BASE_TFIDF = joblib.load("models/tfidf_vec.pkl")
BASE_LR = joblib.load("models/tfidf_lr.pkl")

st.set_page_config(page_title="Baseline Sentiment", page_icon="✅", layout="centered")
st.title("✅ Baseline Sentiment Checker")

text = st.text_area("Paste a movie review:", height=180)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please paste a review first.")
    else:
        X = BASE_TFIDF.transform([text])
        proba = float(BASE_LR.predict_proba(X)[0, 1])
        pred = "POSITIVE" if proba >= 0.5 else "NEGATIVE"
        st.metric("Prediction", pred, f"p(pos)={proba:.3f}")
