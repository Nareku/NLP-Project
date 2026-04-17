# NLP Project
# 📰 Detecting Hoax News in Bahasa Indonesia using IndoBERT

## 📌 Project Based Learning – Natural Language Processing

This project focuses on building an NLP system to detect **hoax news in Indonesian language** using Transformer-based models, specifically **IndoBERT** and **mBERT**, and comparing their performance.

---

## 📖 Chapter 1: Introduction

### 🔹 Background
Hoaxes and misinformation spread rapidly in Indonesia through social media and news platforms. This project aims to build an NLP system that helps detect fake news automatically using deep learning models.

### 🔹 Problem Statement
Hoax detection is treated as a **binary classification task**:
- 0 → Non-Hoax  
- 1 → Hoax  

Challenges:
- Imbalanced datasets  
- Contextual language understanding  
- Complex Indonesian linguistic structure  

### 🔹 Proposed Solution
We use **Transformer-based models**:
- IndoBERT (language-specific model)
- mBERT (multilingual model)

We fine-tune both models and compare performance.

---

## 📚 Chapter 2: Related Work

- Traditional ML: Naive Bayes, SVM, Logistic Regression (TF-IDF)
- Deep Learning: CNN, RNN
- Transformer Models: BERT, IndoBERT

📌 Key Insight:
- Transformer models outperform traditional ML.
- Language-specific models (IndoBERT) perform better for Indonesian NLP tasks.

---

## 📊 Chapter 3: Dataset

### 🔹 Dataset Source
- Kaggle: Indonesia False News (Hoax) Dataset

### 🔹 Dataset Description
- Two files:
  - `Data_latih.csv` (training set)
  - `Data_uji.csv` (test set)
- Labels:
  - 0 = Non-Hoax  
  - 1 = Hoax  

### 🔹 Key Features
- Real Indonesian news data  
- Binary classification  
- Balanced using upsampling  

---

## ⚙️ Chapter 4: Preprocessing

### 🔹 Text Cleaning
- Lowercasing text
- Removing URLs
- Removing punctuation
- Removing stopwords (Sastrawi)
- Removing extra spaces

### 🔹 Tokenization
- IndoBERT tokenizer
- mBERT tokenizer
- Max length = 128 tokens
- Padding & truncation applied

### 🔹 Output Format
- Input IDs
- Attention masks
- Labels

---

## 🧠 Chapter 5: Results & Analysis

### 🔹 Experimental Setup
- Epochs: 2  
- Batch size: 16  
- Learning rate: 2e-5  
- Optimizer: AdamW  
- Framework: PyTorch + HuggingFace  

---

## 📈 Model Performance

### 🟢 IndoBERT
- Accuracy: **99%**
- Precision (Hoax): **1.00**
- Recall (Hoax): **0.98**

📌 Key Insight:
- Very strong at detecting hoax news
- Minimal false negatives

---

### 🔵 mBERT
- Accuracy: **92%**
- Precision: 0.88
- Recall (Hoax): 0.87

📌 Key Insight:
- Good general performance
- Misses more hoax cases than IndoBERT

---

## 📊 Comparison Summary

| Model     | Accuracy | Hoax Recall | Performance |
|----------|----------|-------------|-------------|
| IndoBERT | 99%      | 0.98        | ⭐ Best |
| mBERT    | 92%      | 0.87        | Good |

---

## 📌 Key Findings

- IndoBERT performs better for Indonesian text
- Multilingual models lose language-specific nuance
- False negatives are critical in hoax detection
- Balanced dataset improves fairness

---

## 🧾 Chapter 6: Conclusion

- IndoBERT significantly outperforms mBERT
- Language-specific pretraining is more effective
- System is suitable for real-world hoax detection
- NLP pipelines improve misinformation filtering

---

## 🚀 Future Work

- Add social media datasets (Twitter, TikTok, etc.)
- Handle sarcasm and implicit hoaxes
- Deploy as web or mobile app
- Real-time hoax detection system

---

## 📚 References

- Kaggle Dataset  
- IndoBERT (EMNLP 2020)  
- BERT (Devlin et al., 2019)  
- HuggingFace Transformers  
- PyTorch  
- Sastrawi NLP tools  

---

## 📷 Results Preview

- Training graphs  
- Confusion matrix  
- Prediction examples  

---

## ⭐ Tools Used
- Python 🐍  
- PyTorch  
- HuggingFace Transformers  
- IndoBERT  
- mBERT  
- Google Colab / GPU  

---


