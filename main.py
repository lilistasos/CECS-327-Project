# libraries need for this project:

import ray
from tasks import analyze_sentiment
import pandas
import matplotlib
import requests
import json

ray.init()

with open("data/reviews.json") as f:
    reviews = json.load(f)

# Run Ray tasks
futures = [analyze_sentiment.remote(r["text"]) for r in reviews]
results = ray.get(futures)

# Save results
with open("results/sentiment_output.json", "w") as f:
    json.dump(results, f, indent=2)

