
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
