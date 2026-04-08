import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="ScamWatch 🔍", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Inter:wght@400;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown("""
    <style>
      [data-testid="stAppViewContainer"] {
        background-color: #e9d1b9 !important;
    }

    [data-testid="stHeader"] {
        background-color: #e9d1b9 !important;
    }

    body {
        background-color: #e9d1b9 !important;
    }
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Inter:wght@400;600&display=swap');

    html, body, [class*="css"] {
        background-color: #e9d1b9 !important;
        color: #3d2b1f !important;
        font-family: 'Inter', sans-serif;
    }

    .pixel-title {
        font-family: 'Press Start 2P', monospace;
        font-size: 42px;
        color: #634c44;
        text-align: center;
        margin-bottom: 6px;
        line-height: 1.6;
    }

    .subtitle {
        text-align: center;
        color: #7a5c4f;
        font-size: 14px;
        margin-bottom: 30px;
        font-family: 'Inter', sans-serif;
    }

    stTextArea textarea {
        background-color: #e9d1b9 !important;
        color: #3d2b1f !important;
        border: 2px solid #c4a882 !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
    }

    .stButton > button {
        background-color: #634c44 !important;
        color: #f5e6d8 !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Press Start 2P', monospace !important;
        font-size: 11px !important;
        padding: 14px !important;
        width: 100% !important;
        transition: 0.2s !important;
    }

    .stButton > button:hover {
        background-color: #4e3a33 !important;
        transform: scale(1.02) !important;
    }

    [data-testid="metric-container"] {
        background-color: #f5e6d8;
        border: 2px solid #c4a882;
        border-radius: 12px;
        padding: 12px;
        text-align: center;
    }

    [data-testid="stMetricLabel"] {
        font-size: 11px !important;
        color: #7a5c4f !important;
        font-family: 'Press Start 2P', monospace !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 13px !important;
        color: #634c44 !important;
        font-weight: 600 !important;
    }

    .stAlert {
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
    }

    hr {
        border-color: #c4a882 !important;
    }

    .footer {
        text-align: center;
        color: #a07c6a;
        font-size: 11px;
        font-family: 'Press Start 2P', monospace;
        margin-top: 10px;
        line-height: 2;
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

st.markdown('<div class="pixel-title">🔍 ScamWatch</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">paste a job posting. find out if its a trap.</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.metric("dataset", "3,000+ posts")
with col2:
    st.metric("accuracy", "95%+")
with col3:
    st.metric("model", "random forest")

st.divider()

st.markdown("**📋 job description**")
job_text = st.text_area("", height=180, placeholder="paste the posting here... sketchy ones welcome 👀")

if st.button("⚡ analyze"):
    if job_text.strip() == "":
        st.warning("paste something first!")
    else:
        with st.spinner("checking..."):
            model, vectorizer = train_model()
            transformed = vectorizer.transform([job_text])
            result = model.predict(transformed)[0]
            prob = model.predict_proba(transformed)[0]

        if result == 1:
            st.error(f"🚨 yeah that's a scam. ({prob[1]*100:.1f}% sure)")
        else:
            st.success(f"✅ looks legit! ({prob[0]*100:.1f}% sure)")

st.divider()
st.markdown('<div class="footer">built by a first year who got tired of fake internships 🍂</div>', unsafe_allow_html=True)
