import gradio as gr
import torch
import numpy as np
import re
import string
from transformers import BertTokenizer, BertForSequenceClassification
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ======================================================
# 1. SETUP PREPROCESSING (Must match tes.ipynb!)
# ======================================================
factory = StopWordRemoverFactory()
stopwords = factory.create_stop_word_remover()

def clean_text(text):
    # 1. Lowercase
    text = str(text).lower()
    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    # 3. Remove Punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 4. Remove Stopwords
    text = stopwords.remove(text)
    return text.strip()

# ======================================================
# 2. LOAD MODEL
# ======================================================
MODEL_PATH = "balanced_hoax_detector"

print("Loading model... (If this fails, your model isn't trained yet!)")
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    print("✅ Model loaded successfully!")
except OSError:
    print("❌ ERROR: Could not find the folder 'balanced_hoax_detector'.")
    print("   Please run your notebook (tes.ipynb) to train and save the model first.")
    exit()

# ======================================================
# 3. PREDICTION FUNCTION
# ======================================================
def predict_hoax(text):
    
    cleaned = clean_text(text)
    print(f"\n[Input Raw] {text}")
    print(f"[Cleaned]   {cleaned}")

    if not cleaned:
        return {"Error": "Text is empty after cleaning."}

    encoded = tokenizer(
        cleaned, 
        truncation=True, 
        padding='max_length', 
        max_length=128, 
        return_tensors='pt'
    )
    
    with torch.no_grad():
        output = model(**encoded)
    
    logits = output.logits
    proba = torch.softmax(logits, dim=1).numpy()[0]
    
    return {
        "Non-Hoax ✅": float(proba[1]),
        "HOAX ❌": float(proba[0])
    }

# ======================================================
# 4. BUILD INTERFACE
# ======================================================
demo = gr.Interface(
    fn=predict_hoax,
    inputs=gr.Textbox(
        lines=4, 
        placeholder="Tulis berita di sini (Contoh: Vaksin itu aman...)", 
        label="Masukkan Berita"
    ),
    outputs=gr.Label(num_top_classes=2, label="Hasil Deteksi"),
    title="📰 IndoBERT Hoax Detector",
    description="Deteksi berita Hoax vs Non-Hoax menggunakan model IndoBERT.",
    examples=[
        ["Vaksin COVID-19 mengandung microchip pelacak."],
        ["Pemerintah resmi memulai vaksinasi tahap kedua hari ini."]
    ],
    theme="default"
)

if __name__ == "__main__":
    demo.launch()