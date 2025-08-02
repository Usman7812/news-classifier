import streamlit as st
import joblib

# Load files
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
mlb = joblib.load("mlb.pkl")

st.title("📰 News Category Classifier")

st.write("Enter a headline and excerpt below, and this app will predict the news categories.")

headline = st.text_input("News Headline")
excerpt = st.text_area("News Excerpt")

threshold = 0.5

if st.button("Predict"):
    if headline and excerpt:
        text = headline + ". " + excerpt
        vec = vectorizer.transform([text])
        proba = model.predict_proba(vec)[0]
        results = [(mlb.classes_[i], proba[i]) for i in range(len(proba)) if proba[i] >= threshold]
        results.sort(key=lambda x: x[1], reverse=True)

        if results:
            st.subheader("📊 Predicted Categories:")
            for cat, score in results:
                st.markdown(f"- **{cat}** ({score:.2f})")
        else:
            st.warning("No categories passed the threshold.")
    else:
        st.error("Please fill both fields.")
