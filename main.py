# libraries need for this project:

import ray
from tasks import analyze_sentiment
import pandas as pd
import matplotlib
import requests
import json
import os
import time
from transformers import pipeline
import random

ray.init()

# Read the CSV file
df = pd.read_csv("data/DisneylandReviews.csv", encoding="latin1").head(25)
# Adjust 'Review_Text' to the actual column name for review text
reviews = df['Review_Text'].dropna().tolist()

# Inspect the columns to find the review text column
print(df.columns)

# Run Ray tasks
futures_hf = [analyze_sentiment.remote(r) for r in reviews]
results_hf = ray.get(futures_hf)

# Save results
os.makedirs("results", exist_ok=True)
with open("results/sentiment_output_hf.json", "w") as f:
    json.dump(results_hf, f, indent=2)

print("Sentiment analysis complete! Results saved to results/sentiment_output_hf.json")

# Hugging Face
start = time.time()
futures_hf = [analyze_sentiment.remote(r) for r in reviews]
results_hf = ray.get(futures_hf)
print(f"Hugging Face time: {time.time() - start:.2f} seconds")

# After sentiment analysis and saving results, add summarization

# Summarize a random sample of 50 reviews into one paragraph
sample_size = 50
sampled_reviews = random.sample(reviews, min(sample_size, len(reviews)))

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Combine sampled reviews into one string (or chunk if too long)
max_chunk_length = 1024  # BART can handle up to 1024 tokens
all_reviews = " ".join(sampled_reviews)

# If too long, split into chunks
chunks = []
while len(all_reviews) > 0:
    chunk = all_reviews[:max_chunk_length]
    last_period = chunk.rfind('.')
    if last_period != -1 and last_period > 200:
        chunk = chunk[:last_period+1]
    chunks.append(chunk)
    all_reviews = all_reviews[len(chunk):]

# Summarize each chunk
summaries = [summarizer(chunk, max_length=60, min_length=20, do_sample=False)[0]['summary_text'] for chunk in chunks]

# If more than one summary, summarize the summaries
if len(summaries) > 1:
    final_summary = summarizer(" ".join(summaries), max_length=60, min_length=20, do_sample=False)[0]['summary_text']
else:
    final_summary = summaries[0]

print("\nOverall Summary of 50 Random Reviews:")
print(final_summary)

# Save the summary to a file
with open("results/summary_hf.txt", "w") as f:
    f.write(final_summary + "\n")