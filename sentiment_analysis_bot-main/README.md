
# 🎬 Sentiment Analysis Bot — IMDb Reviews  
### Classical ML vs Deep Learning Proxy vs GPT-based LLM Prompting  
Built with **Python**, **scikit-learn**, **TensorFlow/Keras**, **OpenAI GPT**, and **Streamlit**

---

## 📌 1. Overview

This project implements an interactive **Sentiment Analysis Bot** that classifies IMDb movie reviews as **Positive** or **Negative** using three techniques:

1. **Baseline ML Model:** TF-IDF + Logistic Regression  
2. **Deep Learning Proxy Model:** Naive Bayes classifier trained on TF-IDF features  
3. **LLM Prompting Model:** GPT (Zero-shot, Few-shot, Chain-of-Thought)

The purpose of this project is to compare the **accuracy**, **speed**, **explainability**, and **practical usability** of classical ML models versus modern Large Language Models (LLMs) for long, nuanced text classification.

---

## 🎯 2. Problem Statement

IMDb movie reviews are often long, emotional, and context-heavy.  
Classical ML models struggle with deep semantics, while deep learning requires significant computing resources. LLMs offer strong reasoning capabilities but may introduce latency and cost.

The problem addressed in this project:

> **Which approach—traditional ML, deep-learning proxy, or LLM prompting—delivers the best combination of accuracy, efficiency, and real-world usability for sentiment analysis?**

---

## 📂 3. Dataset Description

**Dataset:** IMDb Large Movie Review Dataset  
**Link:** https://ai.stanford.edu/~amaas/data/sentiment/  
**Size:** 50,000 labeled reviews  
**Attributes:**
- `review_text` — Complete IMDb movie review  
- `sentiment` — `positive` or `negative`

⚠️ *The dataset is not included in this repository due to size limits.*  

---

## 🧠 4. Methodology

### **Step 1 — Preprocessing**
- Remove HTML tags  
- Lowercase text  
- Tokenization  
- TF-IDF vectorization  

### **Step 2 — Models Used**
#### **1. Baseline Model (TF-IDF + LR)**
Fast, efficient, interpretable classical ML pipeline.

#### **2. Deep Learning Proxy (NB Model)**
A lightweight probabilistic model simulating deeper feature interactions.

#### **3. LLM Prompting (GPT)**
- Zero-shot  
- Few-shot  
- Chain-of-Thought (CoT) reasoning  
Used for contextual, semantic-rich reasoning.

### **Step 3 — Evaluation**
Metrics compared:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Latency  
- Cost estimation (for LLM)

### **Step 4 — Streamlit App**
Fully interactive application offering:

- Review input box  
- Predictions from all three models  
- Confidence bars  
- Chain-of-Thought explanation  
- Comparison charts  
- Prediction history  
- Chatbot panel  
- CSV export functionality  

---

---

## 📊 5. Evaluation Results

### **Accuracy Comparison**
| Model | Accuracy |
|-------|----------|
| Baseline (TF-IDF + LR) | **0.88** |
| Deep Proxy (NB) | **0.83** |
| LLM (GPT) | **0.97** |

### **Precision / Recall / F1-score**
LLMs generally outperform other models on all metrics.

Confusion matrices and additional charts are available inside the Streamlit app.

---







