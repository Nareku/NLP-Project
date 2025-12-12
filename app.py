import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import base64
import numpy as np

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="IndoBERT Hoax Detector",
    page_icon="📰",
    layout="centered",
)

# ======================================================
# CUSTOM CSS (Modern UI + Transitions)
# ======================================================
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #f0f4ff 0%, #d9e4ff 100%);
    font-family: 'Segoe UI', sans-serif;
}

/* Card */
.stCard {
    background: #ffffff80;
    backdrop-filter: blur(12px);
    border-radius: 18px;
    padding: 25px;
    margin-top: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.10);
    animation: fadeIn 1s ease;
}

/* Fade-in Animation */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Result Badge */
.hoax {
    padding: 12px 20px;
    border-radius: 12px;
    font-weight: 700;
    background: #ffcccc;
    color: #b30000;
    text-align: center;
}

.nonhoax {
    padding: 12px 20px;
    border-radius: 12px;
    font-weight: 700;
    background: #ccffcc;
    color: #006600;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL & TOKENIZER
# ======================================================
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("balanced_hoax_detector")
    model = BertForSequenceClassification.from_pretrained("balanced_hoax_detector")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ======================================================
# PREDICTION FUNCTION
# ======================================================
def predict_hoax(text):
    encoded = tokenizer(
        text, truncation=True, padding='max_length', max_length=128, return_tensors='pt'
    )
    with torch.no_grad():
        output = model(**encoded)
    logits = output.logits
    proba = torch.softmax(logits, dim=1).numpy()[0]
    label = np.argmax(proba)

    return label, float(proba[label])

# ======================================================
# HEADER
# ======================================================
st.markdown("<h1 style='text-align:center;'>📰 IndoBERT Hoax Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Deteksi berita Hoax atau Non-Hoax menggunakan model IndoBERT</p>", unsafe_allow_html=True)

# ======================================================
# INPUT SECTION
# ======================================================
st.markdown("<div class='stCard'>", unsafe_allow_html=True)
st.subheader("📝 Masukkan Teks untuk Deteksi Hoax")

text_input = st.text_area("Tulis atau paste berita / narasi di sini...", height=180)

col1, col2 = st.columns(2)

with col1:
    submit_btn = st.button("🚀 Deteksi Sekarang")

with col2:
    csv_file = st.file_uploader("Atau upload file CSV", type="csv")

st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# RESULT: SINGLE TEXT
# ======================================================
if submit_btn and text_input.strip() != "":
    label, confidence = predict_hoax(text_input)

    st.markdown("<div class='stCard'>", unsafe_allow_html=True)
    st.subheader("🔍 Hasil Deteksi")

    if label == 1:
        st.markdown("<div class='hoax'>HOAX ❌</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='nonhoax'>NON-HOAX ✅</div>", unsafe_allow_html=True)

    st.write(f"**Confidence: {confidence:.4f}**")
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# RESULT: CSV BATCH PREDICTION
# ======================================================
if csv_file is not None:
    df = pd.read_csv(csv_file)

    if "text" not in df.columns:
        st.error("CSV harus memiliki kolom bernama **text**")
    else:
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.subheader("📄 Hasil Deteksi dari File CSV")

        predictions = []
        confidences = []

        for txt in df["text"]:
            label, prob = predict_hoax(str(txt))
            predictions.append("Hoax" if label == 1 else "Non-Hoax")
            confidences.append(prob)

        df["Prediction"] = predictions
        df["Confidence"] = confidences

        st.dataframe(df)

        csv_output = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Hasil CSV",
            csv_output,
            "hasil_prediksi_hoax.csv",
            "text/csv"
        )

        st.markdown("</div>", unsafe_allow_html=True)
