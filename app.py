import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

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

st.title("🔍 ScamWatch")
st.subheader("Fake Job Posting Detector for College Students")
st.write("Paste any job description below and find out if it's legit or a scam.")

job_text = st.text_area("Job Description", height=200, placeholder="Paste the job posting here...")

if st.button("Analyze"):
    if job_text.strip() == "":
        st.warning("Please paste a job description first!")
    else:
        model, vectorizer = train_model()
        transformed = vectorizer.transform([job_text])
        result = model.predict(transformed)[0]
        probability = model.predict_proba(transformed)[0]
        
        if result == 1:
            st.error(f"🚨 RED FLAG — Looks Fake! (Confidence: {probability[1]*100:.1f}%)")
        else:
            st.success(f"✅ Looks Legit! (Confidence: {probability[0]*100:.1f}%)")
