import streamlit as st
import joblib

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        header[data-testid="stHeader"] {display: none !important;}
        footer {visibility: hidden;}

        [data-testid="block-container"],
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 1rem !important;
            margin-top: 0 !important;
        }

        [data-testid="stAppViewContainer"] > section > div:first-child {
            padding-top: 0 !important;
        }

        h1 {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        .result-real {
            background-color: #d4f5e2;
            border-left: 5px solid #00a844;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            font-size: 1.1rem;
            color: #006b2b;
            font-weight: 600;
            box-shadow: 0 2px 4px rgba(0,168,68,0.1);
        }

        .result-fake {
            background-color: #fde8e8;
            border-left: 5px solid #e53935;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            font-size: 1.1rem;
            color: #b71c1c;
            font-weight: 600;
            box-shadow: 0 2px 4px rgba(229,57,53,0.1);
        }

        .stats-row {
            font-size: 0.88rem;
            text-align: center;
            margin-top: 0.8rem;
            color: #555555;
        }

        .custom-footer {
            text-align: center;
            font-size: 0.78rem;
            margin-top: 2.5rem;
            color: #aaaaaa;
        }
    </style>
""", unsafe_allow_html=True)

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

st.markdown("# 🔍 Fake News Detector")
st.markdown("##### AI-powered news authenticity checker")
st.divider()

# PLACEHOLDER FOR RESULT - Shows at top
result_placeholder = st.empty()

inputn = st.text_area(
    "Paste your news article here:",
    placeholder="Enter the full news article text...",
    height=180
)

# Example section
with st.expander("📌 See Example"):
    st.write("**Example Real News:** Scientists discover new species...")
    st.write("**Example Fake News:** Celebrity reveals shocking secret wealth hack...")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    check = st.button("🔎 Analyze News", use_container_width=True)

if check:
    if inputn.strip():
        with st.spinner("Analyzing article..."):
            transform_input = vectorizer.transform([inputn])
            prediction = model.predict(transform_input)
            confidence = model.predict_proba(transform_input)[0]
            score = round(max(confidence) * 100, 1)

        # RESULT DIKHAO PLACEHOLDER ME (TOP PAR)
        with result_placeholder.container():
            if prediction[0] == 1:
                st.markdown(
                    f'<div class="result-real">✅ This news appears to be <strong>REAL</strong> &nbsp;|&nbsp; Confidence: {score}%</div>',
                    unsafe_allow_html=True
                )
                st.progress(score/100, text=f"Authenticity Score: {score}%")
            else:
                st.markdown(
                    f'<div class="result-fake">❌ This news appears to be <strong>FAKE</strong> &nbsp;|&nbsp; Confidence: {score}%</div>',
                    unsafe_allow_html=True
                )
                st.progress((100-score)/100, text=f"Suspicion Score: {100-score}%")

            st.markdown(
                f'<div class="stats-row">📝 Words: <strong>{len(inputn.split())}</strong> &nbsp;|&nbsp; Characters: <strong>{len(inputn)}</strong></div>',
                unsafe_allow_html=True
            )
            
            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Check Another", use_container_width=True):
                    result_placeholder.empty()
                    st.rerun()
            with col2:
                if st.button("📋 Clear", use_container_width=True):
                    result_placeholder.empty()
                    st.rerun()
    else:
        st.warning("⚠️ Please enter some text to analyze.")

st.markdown(
    '<div class="custom-footer">Built with Streamlit & Scikit-learn • Logistic Regression Model</div>',
    unsafe_allow_html=True
)