from datasets import load_dataset
import json
import os

os.makedirs("data", exist_ok=True)

# Load the English subset of Amazon Reviews
dataset = load_dataset("amazon_reviews_multi", "en")

# Select a small subset (100 samples)
sample = dataset["train"].select(range(100))

# Create list of dicts
output = [{"text": x["review_body"], "stars": x["stars"]} for x in sample]

# Save to JSON
with open("data/reviews.json", "w") as f:
    json.dump(output, f, indent=2)

print("reviews.json created with", len(output), "entries.")
