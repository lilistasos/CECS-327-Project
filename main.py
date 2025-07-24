# libraries need for this project:

import ray
from tasks import analyze_sentiment, analyze_sentiment_chatgpt
import pandas as pd
import matplotlib
import requests
import json
import os
import time

ray.init()

print("API KEY:", os.getenv("OPENAI_API_KEY"))

# Read the CSV file
df = pd.read_csv("data/DisneylandReviews.csv", encoding="latin1").head(10)
# Adjust 'Review_Text' to the actual column name for review text
reviews = df['Review_Text'].dropna().tolist()

# Inspect the columns to find the review text column
print(df.columns)

# Run Ray tasks
futures_hf = [analyze_sentiment.remote(r) for r in reviews]
results_hf = ray.get(futures_hf)

futures_gpt = [analyze_sentiment_chatgpt.remote(r) for r in reviews]
results_gpt = ray.get(futures_gpt)

# Combine results for comparison
comparison_results = []
for r, hf, gpt in zip(reviews, results_hf, results_gpt):
    comparison_results.append({
        "text": r,
        "sentiment_HuggingFace": hf["sentiment"] if isinstance(hf, dict) and "sentiment" in hf else hf,
        "sentiment_OpenAI": gpt["sentiment"] if isinstance(gpt, dict) and "sentiment" in gpt else gpt
    })

os.makedirs("results", exist_ok=True)
with open("results/sentiment_output_comparison.json", "w") as f:
    json.dump(comparison_results, f, indent=2)

print("Sentiment analysis complete! Results saved to results/sentiment_output_comparison.json")

# Hugging Face
start = time.time()
futures_hf = [analyze_sentiment.remote(r) for r in reviews]
results_hf = ray.get(futures_hf)
print(f"Hugging Face time: {time.time() - start:.2f} seconds")

# ChatGPT
start = time.time()
futures_gpt = [analyze_sentiment_chatgpt.remote(r) for r in reviews]
results_gpt = ray.get(futures_gpt)
print(f"ChatGPT time: {time.time() - start:.2f} seconds")

# Save both results
with open("results/sentiment_output_hf.json", "w") as f:
    json.dump(results_hf, f, indent=2)
with open("results/sentiment_output_gpt.json", "w") as f:
    json.dump(results_gpt, f, indent=2)