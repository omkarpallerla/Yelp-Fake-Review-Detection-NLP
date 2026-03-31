# ⭐ Yelp Fake Review Detection System (NLP + Databricks Pipeline)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=python&logoColor=white)

> **NLP fraud detection engine achieving 99% F1 — architected for real-time scoring via a Databricks Structured Streaming pipeline.**

---

## 📌 Business Overview

Fake "astroturfing" reviews damage brand reputation and mislead customers. This project builds an automated Fraud Detection Engine that processes unstructured review text to flag suspicious patterns using semantic and behavioral signals.

The scoring model runs inside a **Databricks Structured Streaming pipeline** that processes incoming reviews in near-real-time before they hit the platform.

---

## 🧠 Technical Architecture

```
Yelp Reviews (Kafka/API)
  → Databricks Structured Streaming (Bronze layer)
    → Feature Engineering Pipeline (Silver layer)
      → Isolation Forest (anomaly pre-filter)
        → XGBoost Classifier (supervised scoring)
          → Gold Layer: flagged_reviews → Power BI Moderation Dashboard
```

---

## 📊 Three-Stage Detection

| Stage | Method | Signal |
|-------|--------|--------|
| NLP Features | VADER + TextBlob | Fake reviews cluster at ±1.0 polarity |
| Anomaly Detection | Isolation Forest, One-Class SVM | Outlier review behavior |
| Classification | XGBoost | Trained on anomaly-labeled data |

---

## 📈 Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 99% |
| **F1-Score** | 0.99 |
| **Fake Review Rate** | 8.3% flagged |
| **Moderator Workload Reduction** | 90%+ |
| **Streaming Latency** | < 8 seconds |

---

## 🛠 Tools & Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10, PySpark |
| NLP | NLTK (VADER), TextBlob |
| ML | XGBoost, Isolation Forest, One-Class SVM |
| Streaming | Databricks Structured Streaming |
| BI Output | Delta table → Power BI Moderation Dashboard |

---

## 🚀 How to Run

```bash
git clone https://github.com/omkarpallerla/Yelp-Fake-Review-Detection-NLP.git
cd Yelp-Fake-Review-Detection-NLP
pip install -r requirements.txt
jupyter notebook notebooks/01_EDA_and_Feature_Engineering.ipynb
```

---

<div align="center">
  <sub>Built by <a href="https://github.com/omkarpallerla">Omkar Pallerla</a> · MS Business Analytics, ASU · BI Engineer · Databricks | Azure | GCP Certified</sub>
</div>