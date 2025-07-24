from datasets import load_dataset
import json
import os

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# Load the IMDb reviews dataset (train split)
dataset = load_dataset("imdb", split="train")

# Select the first 1000 reviews (adjust as needed)
subset = dataset.select(range(1000))

# Convert to list of dicts
data = [dict(row) for row in subset]

# Save to JSON
with open("data/reviews.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Saved {len(data)} reviews to data/reviews.json")