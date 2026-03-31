# =============================================================
# Yelp Fake Review Detection — NLP + Streaming Pipeline
# Author: Omkar Pallerla | MS Business Analytics, ASU
# Real-time Databricks Structured Streaming architecture
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

nltk.download('vader_lexicon', quiet=True)
plt.style.use('dark_background')

# ── 1. GENERATE REALISTIC SAMPLE DATA ──────────────────────
np.random.seed(42)
n_real, n_fake = 1800, 200

def gen_review(fake=False):
    if fake:
        templates = [
            "Amazing! Best ever! Highly recommend to everyone!",
            "Absolutely terrible. Never going back. Worst experience.",
            "Perfect! Five stars! Outstanding service!",
            "Horrible! Stay away! Complete disaster!",
            "The best! Amazing! Love it!"
        ]
        return np.random.choice(templates)
    words = ['good', 'okay', 'nice', 'decent', 'fine', 'pretty good', 'not bad',
             'enjoyable', 'satisfactory', 'reasonable', 'adequate', 'pleasant']
    length = np.random.randint(20, 80)
    return ' '.join(np.random.choice(words, size=length))

real_reviews = [gen_review(False) for _ in range(n_real)]
fake_reviews = [gen_review(True)  for _ in range(n_fake)]

df = pd.DataFrame({
    'review_id':   range(n_real + n_fake),
    'text':        real_reviews + fake_reviews,
    'label':       [0] * n_real + [1] * n_fake,
    'date':        pd.date_range('2023-01-01', periods=n_real + n_fake, freq='2H'),
    'business_id': np.random.randint(1, 50, n_real + n_fake),
    'stars':       ([np.random.randint(2, 5) for _ in range(n_real)] +
                    [np.random.choice([1, 5]) for _ in range(n_fake)])
})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ── 2. FEATURE ENGINEERING ──────────────────────────────────
sia = SentimentIntensityAnalyzer()

def extract_features(row):
    text = row['text']
    vader    = sia.polarity_scores(text)
    blob     = TextBlob(text)
    words    = text.split()
    return {
        'vader_compound':    vader['compound'],
        'vader_pos':         vader['pos'],
        'vader_neg':         vader['neg'],
        'vader_neu':         vader['neu'],
        'textblob_polarity': blob.sentiment.polarity,
        'textblob_subjectivity': blob.sentiment.subjectivity,
        'word_count':        len(words),
        'char_count':        len(text),
        'exclamation_count': text.count('!'),
        'cap_ratio':         sum(1 for c in text if c.isupper()) / max(len(text), 1),
        'unique_word_ratio': len(set(words)) / max(len(words), 1),
        'avg_word_length':   np.mean([len(w) for w in words]) if words else 0,
        'stars':             row['stars']
    }

print("Extracting NLP features...")
features = df.apply(extract_features, axis=1)
feature_df = pd.DataFrame(list(features))

# Temporal burst flag
df['hour'] = pd.to_datetime(df['date']).dt.hour
burst_count = df.groupby(['business_id', df['date'].dt.date])['review_id'].count()
df['daily_reviews'] = df.apply(
    lambda r: burst_count.get((r['business_id'], r['date'].date()), 0), axis=1)
feature_df['burst_flag'] = (df['daily_reviews'] > 5).astype(int)

# Extreme sentiment flag
feature_df['extreme_sentiment'] = (
    (feature_df['vader_compound'].abs() == 1.0) |
    (feature_df['word_count'] < 8)
).astype(int)

# ── 3. STAGE 1 — Isolation Forest (unsupervised) ───────────
iso = IsolationForest(contamination=0.1, random_state=42)
feature_df['anomaly_score'] = iso.fit_predict(feature_df)
feature_df['is_anomaly']    = (feature_df['anomaly_score'] == -1).astype(int)
print(f"Isolation Forest flagged: {feature_df['is_anomaly'].sum()} anomalies")

# ── 4. STAGE 2 — XGBoost classifier ────────────────────────
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    feature_df, y, test_size=0.2, stratify=y, random_state=42)

xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                     scale_pos_weight=n_real/n_fake, random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
y_prob = xgb.predict_proba(X_test)[:, 1]

print(f"\nF1 Score: {f1_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

# ── 5. EXPORT FLAGGED REVIEWS ───────────────────────────────
result_df = df.loc[X_test.index].copy()
result_df['fraud_probability'] = y_prob
result_df['flagged']           = y_pred
result_df['anomaly_score']     = feature_df.loc[X_test.index, 'anomaly_score'].values
result_df[result_df['flagged'] == 1].to_csv('outputs/flagged_reviews.csv', index=False)
print(f"Exported: outputs/flagged_reviews.csv ({result_df['flagged'].sum()} flagged)")

# ── 6. DATABRICKS STREAMING PIPELINE STUB ──────────────────
streaming_code = '''
# pipeline/streaming_pipeline.py
# Databricks Structured Streaming — deploy to cluster

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark = SparkSession.builder.appName("FakeReviewDetection").getOrCreate()

# Read from Kafka / Pub/Sub topic
reviews_stream = (spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "your-broker:9092")
    .option("subscribe", "yelp-reviews")
    .load())

# Parse JSON
schema = StructType([
    StructField("review_id", StringType()),
    StructField("text", StringType()),
    StructField("stars", IntegerType()),
    StructField("business_id", StringType()),
    StructField("timestamp", TimestampType())
])

parsed = reviews_stream.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

# Feature extraction UDF
@udf(returnType=FloatType())
def get_word_count(text):
    return float(len(text.split())) if text else 0.0

@udf(returnType=IntegerType())
def get_exclamations(text):
    return text.count('!') if text else 0

enriched = parsed.withColumn("word_count", get_word_count("text"))
                 .withColumn("exclamation_count", get_exclamations("text"))
                 .withColumn("processed_at", current_timestamp())

# Write to Delta Lake gold layer
query = (enriched.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", "/delta/checkpoints/reviews")
    .table("gold.flagged_reviews"))

query.awaitTermination()
'''
with open('pipeline/streaming_pipeline.py', 'w') as f:
    f.write(streaming_code)

# ── 7. VISUALIZATIONS ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('#0d1117')

# Sentiment distribution
ax = axes[0, 0]
real_sent = feature_df[y == 0]['vader_compound']
fake_sent = feature_df[y == 1]['vader_compound']
ax.hist(real_sent, bins=30, alpha=0.7, color='#06d6a0', label='Real Reviews')
ax.hist(fake_sent, bins=30, alpha=0.7, color='#ef4444', label='Fake Reviews')
ax.axvline(1.0,  color='red',   linestyle='--', alpha=0.5)
ax.axvline(-1.0, color='red',   linestyle='--', alpha=0.5)
ax.set_xlabel('VADER Sentiment Score')
ax.set_title('Sentiment Distribution: Real vs Fake', color='white', pad=12)
ax.legend()

# Word count distribution
ax = axes[0, 1]
ax.hist(feature_df[y == 0]['word_count'], bins=30, alpha=0.7, color='#4f9cf9', label='Real')
ax.hist(feature_df[y == 1]['word_count'], bins=30, alpha=0.7, color='#f59e0b', label='Fake')
ax.set_xlabel('Word Count')
ax.set_title('Review Length: Real vs Fake', color='white', pad=12)
ax.legend()

# Confusion matrix
ax = axes[1, 0]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
ax.set_title('Confusion Matrix — XGBoost', color='white', pad=12)

# Feature importance
ax = axes[1, 1]
feat_imp = pd.Series(xgb.feature_importances_, index=feature_df.columns).sort_values(ascending=False).head(8)
ax.barh(feat_imp.index[::-1], feat_imp.values[::-1], color='#7c3aed')
ax.set_title('Top Feature Importances', color='white', pad=12)

plt.tight_layout()
plt.savefig('outputs/fake_review_analysis.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print("Saved: outputs/fake_review_analysis.png")
plt.show()
