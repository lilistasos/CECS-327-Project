import ray
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

@ray.remote
def analyze_sentiment(text):
    result = sentiment_pipeline(text)
    return {"text": text, "sentiment": result[0]["label"]}