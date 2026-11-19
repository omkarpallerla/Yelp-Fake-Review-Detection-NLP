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

## 📂 Repository Structure

```
Yelp-Fake-Review-Detection-NLP/
├── yelp_fake_review_detection.py
├── notebooks/Yelp_NLP_Fake_Detection.ipynb
├── pipeline/streaming_pipeline.py   # generated Databricks Structured Streaming job
├── outputs/                         # flagged_reviews.csv, fake_review_analysis.png
├── analytics_dashboard.html
└── requirements.txt
```

---

## 🚀 How to Run

```bash
git clone https://github.com/omkarpallerla/Yelp-Fake-Review-Detection-NLP.git
cd Yelp-Fake-Review-Detection-NLP
pip install -r requirements.txt
jupyter notebook yelp_fake_review_detection.py   # or: jupyter notebook notebooks/Yelp_NLP_Fake_Detection.ipynb
```

---


## 🧪 Data & Reproducibility

The script runs on **synthetic, template-generated reviews** (2,000 rows, 10% fake) rather than the Yelp dataset, so the classes are trivially separable and it scores F1 ≈ 1.0 - that is a pipeline demonstration, not a benchmark. The 99% F1 / 8.3% flag rate / latency figures above come from the original project run. `pipeline/streaming_pipeline.py` is a generated Databricks job template and has not been deployed from this repo.

---

<div align="center">
  <sub><a href="https://github.com/omkarpallerla">Omkar Pallerla</a> · MS Business Analytics, ASU · Business Intelligence Analyst · Google Cloud | Azure | Databricks | Snowflake Certified</sub>
</div>