import pickle
from pathlib import Path

import streamlit as st

folder = Path(__file__).resolve().parent / "model_assets"


@st.cache_resource
def load_artifacts():
    with (folder / "final_phishing_model.pkl").open("rb") as f:
        model = pickle.load(f)

    with (folder / "vectorizer.pkl").open("rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


st.set_page_config(page_title="Phishing Email Detector")
st.title("Phishing Email Detector")

with st.form("email_classifier"):
    email_text = st.text_area("Paste the email text", height=220)
    submitted = st.form_submit_button("Classify email", type="primary")

if submitted:
    if email_text.strip():
        model, vectorizer = load_artifacts()
        email_vectorized = vectorizer.transform([email_text])
        prediction = model.predict(email_vectorized)
        if prediction[0] == 1:
            st.warning("Phishing Email 🚨")
        else:
            st.success("Legitimate Email ✅")
    else:
        st.warning("Paste an email before classifying.")
