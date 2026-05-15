<div align="center">

# 🔍 Fake News Detector

### AI-Powered News Authenticity Verification System

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

**Paste any news article. Get an instant AI verdict — Real or Fake.**

🚀 **[Live Demo](https://fake-news-detector-chs.streamlit.app/)** · 📒 **[Research Notebook](https://github.com/AyeshaAndleeb/fake-news-detector-using-machine-learning/blob/main/nlp-based-fake-news-classification.ipynb)**

</div>

## 📖 Overview

**Fake News Detector** is a machine learning–powered web app that analyzes a news article and predicts whether it is **real or fake** with a confidence score. It uses a Logistic Regression model trained on ~39,000 labeled articles, backed by TF-IDF text vectorization, and deployed via Streamlit.

## ✨ Features

| Feature | Description |
| --- | --- |
| 🤖 AI Prediction | Logistic Regression trained on ~39,000 labeled news articles |
| 📊 Confidence Score | Shows how confident the model is (0–100%) |
| ⚡ Instant Results | Result appears above the input — no scrolling needed |
| 📝 Article Stats | Displays word count and character count |
| 🌐 Cloud Deployed | Live on Streamlit Community Cloud |

## 🧠 Model Performance

Trained on the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) — 39,105 articles after deduplication (80/20 split).

| Metric | Score |
| --- | --- |
| Accuracy | 99.39% |
| F1-Score | 99.44% |
| ROC-AUC | 99.95% |

**Pipeline:** TF-IDF (unigrams + bigrams, 100k features) → Logistic Regression (C=10, solver=saga, tuned via 5-fold GridSearchCV)

| File | Purpose |
| --- | --- |
| `vectorizer.jb` | Pre-fitted TF-IDF Vectorizer |
| `lr_model.jb` | Trained Logistic Regression classifier (0 = Fake, 1 = Real) |

## 📓 Research Notebook

Full end-to-end ML pipeline: EDA, preprocessing, word clouds, n-gram analysis, multi-model comparison (LR, NB, LinearSVC, Random Forest), hyperparameter tuning, cross-validation, error analysis, and artifact export.

📒 [View Notebook](https://github.com/AyeshaAndleeb/fake-news-detector-using-machine-learning/blob/main/nlp-based-fake-news-classification.ipynb)

## 🛠 Tech Stack

Python 3.8+ · Streamlit · Scikit-learn · Joblib · NLTK

## 🚀 Quick Start

```bash
git clone https://github.com/AyeshaAndleeb/fake-news-detector-using-machine-learning.git
cd fake-news-detector-using-machine-learning
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## ☁️ Deploy to Streamlit Cloud

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select repo, set main file to `app.py` → Deploy

## 📁 Project Structure

```
├── app.py                                    # Streamlit app
├── lr_model.jb                               # Trained model
├── vectorizer.jb                             # TF-IDF Vectorizer
├── requirements.txt
├── nlp-based-fake-news-classification.ipynb  # Research notebook
└── .streamlit/config.toml                    # Theme config
```

## 🔮 Future Improvements

- Fine-tune BERT / RoBERTa for deeper contextual understanding
- Add URL input to fetch articles directly
- Multilingual support (mBERT, XLM-R)
- Word-level explainability (highlight influential terms)
- REST API with FastAPI + Docker

## 📬 Contact

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-AyeshaAndleeb-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AyeshaAndleeb)

*If you found this useful, consider giving it a ⭐ — it means a lot!*

</div>

<div align="center">Built with ❤️ using Python & Streamlit</div>
