<div align="center">

# 🔍 Fake News Detector

### AI-Powered News Authenticity Verification System

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Deployed on Streamlit](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://share.streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

**Paste any news article. Get an instant AI verdict — Real or Fake.**

[Live Demo](#-live-demo) · [Features](#-features) · [How It Works](#-how-it-works) · [Installation](#-installation) · [Screenshots](#-screenshots)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [The Problem](#-the-problem)
- [Live Demo](#-live-demo)
- [Screenshots](#-screenshots)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [How It Works](#-how-it-works)
- [Model & Backend Logic](#-model--backend-logic)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
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

## 🌐 Live Demo

> 🚀 **Try it now:** [fake-news-detector.streamlit.app](https://fake-news-detector-chs.streamlit.app/)

*Replace the link above with your actual Streamlit deployment URL after deploying.*

---

## 📸 Screenshots

### Home Page
![Home Page](images/home.png)

### Real News Result
![Real News Result](images/result_real.png)

### Fake News Result
![Fake News Result](images/result_fake.png)

> 📁 Add your screenshots inside an `images/` folder and update the paths above.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **AI Prediction** | Logistic Regression model trained on thousands of labeled news articles |
| 📊 **Confidence Score** | Displays how confident the model is in its prediction (0–100%) |
| ⚡ **Instant Results** | Results appear above the input — no scrolling required |
| 🎨 **Clean UI** | Minimal, modern interface built for readability and ease of use |
| 📝 **Article Stats** | Shows word count and character count of the analyzed article |
| 🌐 **Cloud Deployed** | Accessible from any device via Streamlit Cloud |
| 📱 **Responsive** | Works on desktop and mobile browsers |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend / UI** | [Streamlit](https://streamlit.io) |
| **ML Model** | Logistic Regression — [Scikit-learn](https://scikit-learn.org) |
| **Text Vectorization** | TF-IDF Vectorizer — Scikit-learn |
| **Model Serialization** | [Joblib](https://joblib.readthedocs.io) |
| **Language** | Python 3.8+ |
| **Deployment** | Streamlit Community Cloud |

---

## ⚙️ How It Works

```
User pastes article
        │
        ▼
Text is cleaned & vectorized
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

**Label Encoding:**

| Label | Meaning |
|---|---|
| `1` | ✅ Real News |
| `0` | ❌ Fake News |

---

## 📁 Project Structure

```
fake-news-detector/
│
├── app.py                  # Main Streamlit application
├── lr_model.jb             # Trained Logistic Regression model
├── vectorizer.jb           # Fitted TF-IDF Vectorizer
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
├── images/                 # Screenshots for README (add your own)
│   ├── home.png
│   ├── result_real.png
│   └── result_fake.png
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
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
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

- [ ] **Deep Learning Model** — Replace Logistic Regression with BERT or LSTM for higher accuracy
- [ ] **URL Input** — Allow users to paste a news article URL directly
- [ ] **Multi-language Support** — Extend detection to non-English articles
- [ ] **Source Credibility Check** — Cross-reference publisher domain reputation
- [ ] **Explainability** — Highlight which words most influenced the prediction
- [ ] **User Feedback Loop** — Let users flag incorrect predictions to improve the model
- [ ] **Batch Analysis** — Upload a CSV of articles and get bulk predictions

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

Please make sure your code is clean and well-commented before submitting a PR.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

<div align="center">

**Abdullah**

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/YOUR_USERNAME)
[![Email](https://img.shields.io/badge/Email-abdullahscripts%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:abdullahscripts@gmail.com)

*If you found this project useful, consider giving it a ⭐ on GitHub — it means a lot!*

</div>

---

<div align="center">
Built with ❤️ using Python & Streamlit
</div>
