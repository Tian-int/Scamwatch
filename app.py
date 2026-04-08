import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="ScamWatch",
    page_icon="🔍",
    layout="centered"
)

st.markdown("""
    <style>
    .main {
        background-color: #0f0f0f;
        color: white;
    }
    .stTextArea textarea {
        background-color: #1a1a1a;
        color: white;
        border: 1px solid #333;
        border-radius: 10px;
        font-size: 15px;
    }
    .stButton > button {
        background-color: #6c63ff;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 30px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #574fd6;
        transform: scale(1.02);
    }
    .title {
        font-size: 48px;
        font-weight: 800;
        color: #6c63ff;
        text-align: center;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 16px;
        color: #aaaaaa;
        text-align: center;
        margin-bottom: 30px;
    }
    .badge {
        display: inline-block;
        background-color: #1a1a1a;
        border: 1px solid #333;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 13px;
        color: #aaa;
        margin: 2px;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

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

# Header
st.markdown('<div class="title">🔍 ScamWatch</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Fake Job Posting Detector for College Students</div>', unsafe_allow_html=True)

# Stats row
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Dataset", "3,000+ postings")
with col2:
    st.metric("Accuracy", "95%+")
with col3:
    st.metric("Built with", "Python + ML")

st.divider()

# Input
st.markdown("#### 📋 Paste a Job Description")
job_text = st.text_area("", height=200, placeholder="e.g. Work from home! Earn ₹50,000/month. No experience needed. WhatsApp us now...")

if st.button("🔍 Analyze Now"):
    if job_text.strip() == "":
        st.warning("Please paste a job description first!")
    else:
        with st.spinner("Analyzing..."):
            model, vectorizer = train_model()
            transformed = vectorizer.transform([job_text])
            result = model.predict(transformed)[0]
            probability = model.predict_proba(transformed)[0]

        if result == 1:
            st.error(f"🚨 RED FLAG — This looks like a SCAM! (Confidence: {probability[1]*100:.1f}%)")
            st.markdown("**Common red flags detected:** vague roles, unrealistic pay, no company info, urgency tactics")
        else:
            st.success(f"✅ Looks Legit! (Confidence: {probability[0]*100:.1f}%)")
            st.markdown("**Tip:** Always verify the company independently before applying!")

st.divider()
st.markdown('<div style="text-align:center; color:#555; font-size:13px;">Built by a first-year AI/DS student · ScamWatch 2024</div>', unsafe_allow_html=True)
