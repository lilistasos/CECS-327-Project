# libraries need for this project:

import ray
from tasks import analyze_sentiment, analyze_sentiment_api_simple, analyze_sentiment_llm_safe, analyze_sentiment_3class
import pandas as pd
import json
import os
import time
import random

ray.init()

print("API KEY:", os.getenv("OPENAI_API_KEY"))

# Read the CSV file
df = pd.read_csv("data/DisneylandReviews.csv", encoding="latin1").head(100)  # Increased to 100 for better scalability demo
# Adjust 'Review_Text' to the actual column name for review text
reviews = df['Review_Text'].dropna().tolist()

# Inspect the columns to find the review text column
print(df.columns)

# Memory-safe version with five approaches
print("Available models:")
print("1. LLM (DialoGPT-small) - Memory Safe ✅")
print("2. API-based (Twitter RoBERTa) - Memory Safe ✅") 
print("3. Hugging Face 3-class (RoBERTa) - Memory Safe ✅")
print("4. Manual Sentiment Analysis - Memory Safe ✅")
print("5. Hugging Face (DistilBERT) - Memory Safe ✅")

# Run LLM sentiment analysis
print("\nRunning LLM sentiment analysis...")
start = time.time()
futures_llm = [analyze_sentiment_llm_safe.remote(r) for r in reviews]
results_llm = ray.get(futures_llm)
print(f"LLM time: {time.time() - start:.2f} seconds")

# Run API-based sentiment analysis
print("\nRunning API-based sentiment analysis...")
start = time.time()
futures_api = [analyze_sentiment_api_simple.remote(r) for r in reviews]
results_api = ray.get(futures_api)
print(f"API-based time: {time.time() - start:.2f} seconds")

# Run 3-class Hugging Face sentiment analysis
print("\nRunning 3-class Hugging Face sentiment analysis...")
start = time.time()
futures_3class = [analyze_sentiment_3class.remote(r) for r in reviews]
results_3class = ray.get(futures_3class)
print(f"3-class Hugging Face time: {time.time() - start:.2f} seconds")

# Create manual sentiment analysis based on keywords
def manual_sentiment_analysis(text):
    text_lower = text.lower()
    
    # Positive keywords
    positive_words = ['great', 'amazing', 'wonderful', 'fantastic', 'excellent', 'love', 'enjoy', 'good', 'nice', 'fabulous', 'exciting']
    # Negative keywords  
    negative_words = ['terrible', 'awful', 'horrible', 'bad', 'disappointing', 'waste', 'let down', 'worst', 'hate', 'dislike', 'poor']
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        sentiment = "POSITIVE"
    elif negative_count > positive_count:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    
    return {"text": text, "sentiment": sentiment}

# Run manual sentiment analysis
print("\nRunning Manual sentiment analysis...")
start = time.time()
results_manual = [manual_sentiment_analysis(r) for r in reviews]
print(f"Manual analysis time: {time.time() - start:.2f} seconds")

# Run Hugging Face (already working)
print("\nRunning Hugging Face sentiment analysis...")
start = time.time()
futures_hf = [analyze_sentiment.remote(r) for r in reviews]
results_hf = ray.get(futures_hf)
print(f"Hugging Face time: {time.time() - start:.2f} seconds")

# Save results
os.makedirs("results", exist_ok=True)
with open("results/sentiment_output_llm.json", "w") as f:
    json.dump(results_llm, f, indent=2)
with open("results/sentiment_output_api.json", "w") as f:
    json.dump(results_api, f, indent=2)
with open("results/sentiment_output_3class.json", "w") as f:
    json.dump(results_3class, f, indent=2)
with open("results/sentiment_output_manual.json", "w") as f:
    json.dump(results_manual, f, indent=2)
with open("results/sentiment_output_hf.json", "w") as f:
    json.dump(results_hf, f, indent=2)

# Create comparison results
comparison_results = []
for i in range(len(reviews)):
    comparison_results.append({
        "text": reviews[i],
        "sentiment_LLM": results_llm[i]["sentiment"],
        "sentiment_API": results_api[i]["sentiment"],
        "sentiment_3Class": results_3class[i]["sentiment"],
        "sentiment_Manual": results_manual[i]["sentiment"],
        "sentiment_HuggingFace": results_hf[i]["sentiment"]
    })

with open("results/sentiment_output_comparison.json", "w") as f:
    json.dump(comparison_results, f, indent=2)

print("Memory-safe sentiment analysis complete!")

# Print sample comparisons
print("\nSample comparisons:")
for i in range(min(3, len(comparison_results))):
    d = comparison_results[i]
    print(f"Review: {d['text'][:80]}...")
    print(f"  LLM: {d['sentiment_LLM']}")
    print(f"  API: {d['sentiment_API']}")
    print(f"  3-Class: {d['sentiment_3Class']}")
    print(f"  Manual: {d['sentiment_Manual']}")
    print(f"  HuggingFace: {d['sentiment_HuggingFace']}")
    print()

print("\n✅ Success! You now have:")
print("- LLM sentiment analysis (DialoGPT-small)")
print("- API-based sentiment analysis (Cloud-based)")
print("- 3-class Hugging Face sentiment analysis (RoBERTa)")
print("- Manual sentiment analysis (Rule-based)")
print("- Hugging Face sentiment analysis (DistilBERT)")
print("- Performance comparison between all five approaches")
print("- Results saved for visualization and reporting")