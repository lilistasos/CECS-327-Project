# libraries need for this project:

import ray
from tasks import analyze_sentiment, analyze_sentiment_api_simple, analyze_sentiment_llm_safe, analyze_sentiment_3class, analyze_sentiment_manual, analyze_sentiment_vader
import pandas as pd
import json
import os
import time
import random
import psutil
import gc

ray.init()

# Memory monitoring functions
def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def print_memory_status(stage=""):
    """Print current memory usage"""
    memory_mb = get_memory_usage()
    print(f"Memory usage {stage}: {memory_mb:.1f} MB")
    return memory_mb

# Read the CSV file
df = pd.read_csv("data/DisneylandReviews.csv", encoding="latin1").head(1000)  # Increased to 1000 for scalability testing
# Adjust 'Review_Text' to the actual column name for review text
reviews = df['Review_Text'].dropna().tolist()

# Inspect the columns to find the review text column
print(df.columns)
print(f"Dataset size: {len(df)} reviews")
print(f"Reviews to process: {len(reviews)} reviews")
print_memory_status("after loading data")

# Memory-safe version with six approaches
print("Available models:")
print("1. LLM (DialoGPT-small) - Memory Safe")
print("2. API-based (Twitter RoBERTa) - Memory Safe") 
print("3. Hugging Face 3-class (RoBERTa) - Memory Safe")
print("4. Manual Sentiment Analysis - Memory Safe")
print("5. Hugging Face (DistilBERT) - Memory Safe")
print("6. VADER (Valence Aware Dictionary) - Memory Safe")

# Run LLM sentiment analysis
print("\nRunning LLM sentiment analysis...")
llm_memory_before = print_memory_status("before LLM")
start = time.time()
futures_llm = [analyze_sentiment_llm_safe.remote(r) for r in reviews]
results_llm = ray.get(futures_llm)
llm_time = time.time() - start
print(f"LLM time: {llm_time:.2f} seconds")
llm_memory_after = print_memory_status("after LLM")
gc.collect()  # Force garbage collection

# Run API-based sentiment analysis
print("\nRunning API-based sentiment analysis...")
api_memory_before = print_memory_status("before API")
start = time.time()
futures_api = [analyze_sentiment_api_simple.remote(r) for r in reviews]
results_api = ray.get(futures_api)
api_time = time.time() - start
print(f"API-based time: {api_time:.2f} seconds")
api_memory_after = print_memory_status("after API")
gc.collect()

# Run 3-class Hugging Face sentiment analysis
print("\nRunning 3-class Hugging Face sentiment analysis...")
class3_memory_before = print_memory_status("before 3-class")
start = time.time()
futures_3class = [analyze_sentiment_3class.remote(r) for r in reviews]
results_3class = ray.get(futures_3class)
class3_time = time.time() - start
print(f"3-class Hugging Face time: {class3_time:.2f} seconds")
class3_memory_after = print_memory_status("after 3-class")
gc.collect()

# Run manual sentiment analysis
print("\nRunning Manual sentiment analysis...")
manual_memory_before = print_memory_status("before Manual")
start = time.time()
futures_manual = [analyze_sentiment_manual.remote(r) for r in reviews]
results_manual = ray.get(futures_manual)
manual_time = time.time() - start
print(f"Manual analysis time: {manual_time:.6f} seconds ({manual_time/len(reviews):.6f}s per review)")
manual_memory_after = print_memory_status("after Manual")
gc.collect()

# Run Hugging Face (already working)
print("\nRunning Hugging Face sentiment analysis...")
hf_memory_before = print_memory_status("before HuggingFace")
start = time.time()
futures_hf = [analyze_sentiment.remote(r) for r in reviews]
results_hf = ray.get(futures_hf)
hf_time = time.time() - start
print(f"Hugging Face time: {hf_time:.2f} seconds")
hf_memory_after = print_memory_status("after HuggingFace")
gc.collect()

# Run VADER sentiment analysis
print("\nRunning VADER sentiment analysis...")
vader_memory_before = print_memory_status("before VADER")
start = time.time()
futures_vader = [analyze_sentiment_vader.remote(r) for r in reviews]
results_vader = ray.get(futures_vader)
vader_time = time.time() - start
print(f"VADER time: {vader_time:.2f} seconds")
vader_memory_after = print_memory_status("after VADER")
gc.collect()

# Save results with timing data
os.makedirs("results", exist_ok=True)

# Save LLM results with timing and memory
llm_data = {
    "results": results_llm,
    "performance": {
        "total_time": llm_time,
        "time_per_review": llm_time / len(reviews),
        "reviews_processed": len(reviews),
        "memory_before": llm_memory_before,
        "memory_after": llm_memory_after,
        "memory_used": llm_memory_after - llm_memory_before
    }
}
with open("results/sentiment_output_llm.json", "w") as f:
    json.dump(llm_data, f, indent=2)

# Save API results with timing and memory
api_data = {
    "results": results_api,
    "performance": {
        "total_time": api_time,
        "time_per_review": api_time / len(reviews),
        "reviews_processed": len(reviews),
        "memory_before": api_memory_before,
        "memory_after": api_memory_after,
        "memory_used": api_memory_after - api_memory_before
    }
}
with open("results/sentiment_output_api.json", "w") as f:
    json.dump(api_data, f, indent=2)

# Save 3-class results with timing and memory
class3_data = {
    "results": results_3class,
    "performance": {
        "total_time": class3_time,
        "time_per_review": class3_time / len(reviews),
        "reviews_processed": len(reviews),
        "memory_before": class3_memory_before,
        "memory_after": class3_memory_after,
        "memory_used": class3_memory_after - class3_memory_before
    }
}
with open("results/sentiment_output_3class.json", "w") as f:
    json.dump(class3_data, f, indent=2)

# Save manual results with timing and memory
manual_data = {
    "results": results_manual,
    "performance": {
        "total_time": manual_time,
        "time_per_review": manual_time / len(reviews),
        "reviews_processed": len(reviews),
        "memory_before": manual_memory_before,
        "memory_after": manual_memory_after,
        "memory_used": manual_memory_after - manual_memory_before
    }
}
with open("results/sentiment_output_manual.json", "w") as f:
    json.dump(manual_data, f, indent=2)

# Save HuggingFace results with timing and memory
hf_data = {
    "results": results_hf,
    "performance": {
        "total_time": hf_time,
        "time_per_review": hf_time / len(reviews),
        "reviews_processed": len(reviews),
        "memory_before": hf_memory_before,
        "memory_after": hf_memory_after,
        "memory_used": hf_memory_after - hf_memory_before
    }
}
with open("results/sentiment_output_hf.json", "w") as f:
    json.dump(hf_data, f, indent=2)

# Save VADER results with timing and memory
vader_data = {
    "results": results_vader,
    "performance": {
        "total_time": vader_time,
        "time_per_review": vader_time / len(reviews),
        "reviews_processed": len(reviews),
        "memory_before": vader_memory_before,
        "memory_after": vader_memory_after,
        "memory_used": vader_memory_after - vader_memory_before
    }
}
with open("results/sentiment_output_vader.json", "w") as f:
    json.dump(vader_data, f, indent=2)

# Create comparison results
comparison_results = []
for i in range(len(reviews)):
    comparison_results.append({
        "text": reviews[i],
        "sentiment_LLM": results_llm[i]["sentiment"],
        "sentiment_API": results_api[i]["sentiment"],
        "sentiment_3Class": results_3class[i]["sentiment"],
        "sentiment_Manual": results_manual[i]["sentiment"],
        "sentiment_HuggingFace": results_hf[i]["sentiment"],
        "sentiment_VADER": results_vader[i]["sentiment"]
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
    print(f"  VADER: {d['sentiment_VADER']}")
    print()

print("\nSuccess! You now have:")
print("- LLM sentiment analysis (DialoGPT-small)")
print("- API-based sentiment analysis (Cloud-based)")
print("- 3-class Hugging Face sentiment analysis (RoBERTa)")
print("- Manual sentiment analysis (Rule-based)")
print("- Hugging Face sentiment analysis (DistilBERT)")
print("- VADER sentiment analysis (Valence Aware Dictionary)")
print("- Performance comparison between all six approaches")
print("- Results saved for visualization and reporting")