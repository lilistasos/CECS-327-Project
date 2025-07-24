# libraries need for this project:

import ray
from tasks import analyze_sentiment
import pandas as pd
import matplotlib
import requests
import json
import os

ray.init()

# Read the CSV file
df = pd.read_csv("data/DisneylandReviews.csv", encoding="latin1").head(10)
# Adjust 'Review_Text' to the actual column name for review text
reviews = df['Review_Text'].dropna().tolist()

# Inspect the columns to find the review text column
print(df.columns)

# Run Ray tasks
futures = [analyze_sentiment.remote(r) for r in reviews]
results = ray.get(futures)

# Save results
os.makedirs("results", exist_ok=True)
with open("results/sentiment_output.json", "w") as f:
    json.dump(results, f, indent=2)

print("Sentiment analysis complete! Results saved to results/sentiment_output.json")

