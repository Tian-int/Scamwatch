import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="ScamWatch 🔍", layout="centered")

st.title("🔍 ScamWatch")
st.caption("Paste a job posting below and I'll tell you if it's a scam.")

@st.cache_resource
def train_model():
    df = pd.read_csv("data_sample.csv")
    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['requirements'].fillna('')
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['fraudulent']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, vectorizer

job_text = st.text_area("Job Description", height=200, placeholder="Paste the job posting here...")

if st.button("Analyze"):
    if job_text.strip() == "":
        st.warning("Please paste something first!")
    else:
        with st.spinner("Analyzing..."):
            model, vectorizer = train_model()
            transformed = vectorizer.transform([job_text])
            result = model.predict(transformed)[0]
            prob = model.predict_proba(transformed)[0]

        if result == 1:
            st.error(f"🚨 Looks Fake! Confidence: {prob[1]*100:.1f}%")
        else:
            st.success(f"✅ Looks Legit! Confidence: {prob[0]*100:.1f}%")

st.divider()
st.caption("Built with Python · scikit-learn · Streamlit")
