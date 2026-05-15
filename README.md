<div align="center">

# 🔍 Fake News Detector

### AI-Powered News Authenticity Verification System

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Kaggle](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

**Paste any news article. Get an instant AI verdict — Real or Fake.**

[Overview](#-overview) · [Features](#-features) · [How It Works](#-how-it-works) · [Model Details](#-model--backend-logic) · [Installation](#-installation--setup) · [Notebook](#-research-notebook)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [The Problem](#-the-problem)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [How It Works](#-how-it-works)
- [Model & Backend Logic](#-model--backend-logic)
- [Research Notebook](#-research-notebook)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Deployment](#️-deployment-streamlit-cloud)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 📖 Overview

**Fake News Detector** is a machine learning–powered web application that analyzes a news article and predicts whether it is **real or fake** — with a confidence percentage. Built with Streamlit for an instant, interactive experience, it uses a trained Logistic Regression model backed by TF-IDF text vectorization to make fast and reliable predictions.

Whether you're a researcher, student, or just someone who wants to verify what they're reading, this tool gives you an AI-backed second opinion in seconds.

---

## 🚨 The Problem

Misinformation and fake news have become a global crisis. With millions of articles published daily, it is nearly impossible to manually verify the authenticity of every piece of content. This project addresses that challenge by applying Natural Language Processing (NLP) and Machine Learning to automatically detect deceptive or fabricated news content.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **AI Prediction** | Logistic Regression model trained on ~39,000 labeled news articles |
| 📊 **Confidence Score** | Displays how confident the model is in its prediction (0–100%) |
| ⚡ **Instant Results** | Results appear above the input — no scrolling required |
| 🎨 **Clean UI** | Minimal, modern interface built for readability and ease of use |
| 📝 **Article Stats** | Shows word count and character count of the analyzed article |
| 🌐 **Cloud Ready** | One-click deployable to Streamlit Community Cloud |
| 📱 **Responsive** | Works on desktop and mobile browsers |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend / UI** | [Streamlit](https://streamlit.io) |
| **ML Model** | Logistic Regression — [Scikit-learn](https://scikit-learn.org) |
| **Text Vectorization** | TF-IDF Vectorizer (unigrams + bigrams, 100k features) — Scikit-learn |
| **Model Serialization** | [Joblib](https://joblib.readthedocs.io) |
| **Language** | Python 3.8+ |
| **Deployment** | Streamlit Community Cloud |

---

## ⚙️ How It Works

```
User pastes article
        │
        ▼
Text is vectorized
using TF-IDF Vectorizer
        │
        ▼
Feature vector passed to
Logistic Regression model
        │
        ▼
Model outputs prediction
+ probability scores
        │
        ▼
Result displayed instantly:
✅ REAL  or  ❌ FAKE
with Confidence %
```

**Step-by-step:**

1. **Input** — User pastes a news article into the text area.
2. **Vectorization** — The raw text is transformed into numerical features using a pre-fitted TF-IDF Vectorizer.
3. **Prediction** — The feature vector is fed into a trained Logistic Regression classifier.
4. **Output** — The app displays a verdict (Real / Fake) along with the model's confidence percentage and article statistics.

---

## 🧠 Model & Backend Logic

The prediction pipeline consists of two serialized components:

| File | Purpose |
|---|---|
| `vectorizer.jb` | Pre-fitted `TfidfVectorizer` — converts raw text into weighted numerical features |
| `lr_model.jb` | Trained `LogisticRegression` classifier — outputs class label (0 = Fake, 1 = Real) and probability scores |

**Why Logistic Regression?**

Logistic Regression performs exceptionally well on high-dimensional, sparse text data. It is fast, interpretable, and highly effective for binary text classification tasks — making it an ideal choice for this application.

**Model Performance (on held-out test set of 7,821 articles):**

| Metric | Score |
|---|---|
| Accuracy | 99.39% |
| Precision | 99.16% |
| Recall | 99.72% |
| F1-Score | 99.44% |
| ROC-AUC | 99.95% |

**Label Encoding:**

| Label | Meaning |
|---|---|
| `1` | ✅ Real News |
| `0` | ❌ Fake News |

---

## 📓 Research Notebook

The full end-to-end machine learning pipeline is documented in [`nlp-based-fake-news-classification.ipynb`](nlp-based-fake-news-classification.ipynb), covering:

| Step | Description |
|---|---|
| 📥 Data Loading | Loaded and merged ~44,898 articles (Fake.csv + True.csv) |
| 🔍 EDA | Class distribution, article length analysis, subject breakdown |
| 🧹 Preprocessing | Lowercase, URL/HTML removal, lemmatization, stopword removal |
| ☁️ Word Clouds | Visual comparison of fake vs. real news vocabulary |
| 📊 N-gram Analysis | Bigram and trigram frequency analysis by label |
| 🔢 TF-IDF | Unigram + bigram feature extraction (100k features, sublinear TF) |
| 🤖 Model Training | Logistic Regression, Naive Bayes, LinearSVC, Random Forest |
| 🎯 Hyperparameter Tuning | 5-fold GridSearchCV (best: C=10, solver=saga) |
| ✅ Cross-Validation | 5-fold stratified CV — Mean F1: 0.9942 ± low variance |
| 📉 Error Analysis | Investigated misclassified samples and confidence distribution |
| 💾 Artifact Export | Saved model and vectorizer as `.jb` files via Joblib |

**Dataset:** [Fake and Real News Dataset — Clément Bisaillon (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

| Split | Fake | Real | Total |
|---|---|---|---|
| After deduplication | ~18,780 | ~20,325 | 39,105 |
| Train (80%) | — | — | 31,284 |
| Test (20%) | 3,582 | 4,239 | 7,821 |

---

## 📁 Project Structure

```
fake-news-detector/
│
├── app.py                                    # Main Streamlit application
├── lr_model.jb                               # Trained Logistic Regression model
├── vectorizer.jb                             # Fitted TF-IDF Vectorizer
├── requirements.txt                          # Python dependencies
├── nlp-based-fake-news-classification.ipynb  # Full ML research notebook
├── .streamlit/
│   └── config.toml                           # Streamlit theme configuration
├── .gitignore
└── README.md
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/abdullahscripts/fake-news-detector.git
cd fake-news-detector
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

### 5. Open in Browser

```
http://localhost:8501
```

---

## ☁️ Deployment (Streamlit Cloud)

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"** and select your repository.
4. Set the main file path to `app.py`.
5. Click **"Deploy"** — your app will be live in minutes.

---

## 🔮 Future Improvements

- [ ] **Deep Learning Model** — Replace Logistic Regression with BERT or RoBERTa for higher contextual accuracy
- [ ] **URL Input** — Allow users to paste a news article URL directly
- [ ] **Multi-language Support** — Extend detection to non-English articles (mBERT, XLM-R)
- [ ] **Source Credibility Check** — Cross-reference publisher domain reputation
- [ ] **Explainability** — Highlight which words most influenced the prediction
- [ ] **Knowledge Graph Checker** — Cross-reference claims against verified fact databases
- [ ] **User Feedback Loop** — Let users flag incorrect predictions to improve the model
- [ ] **Batch Analysis** — Upload a CSV of articles and get bulk predictions
- [ ] **REST API** — FastAPI + Docker deployment for programmatic access

---

## 🤝 Contributing

Contributions are welcome and appreciated.

```bash
# 1. Fork the repository
# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Commit your changes
git commit -m "Add: your feature description"

# 4. Push to your branch
git push origin feature/your-feature-name

# 5. Open a Pull Request
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

<div align="center">

**Abdullah**

[![GitHub](https://img.shields.io/badge/GitHub-abdullahscripts-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/abdullahscripts)
[![Email](https://img.shields.io/badge/Email-abdullahscripts%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:abdullahscripts@gmail.com)

*If you found this project useful, consider giving it a ⭐ on GitHub — it means a lot!*

</div>

---

<div align="center">
Built with ❤️ using Python & Streamlit
</div>
