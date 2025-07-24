from datasets import load_dataset
import json

dataset = load_dataset("amazon_reviews_multi", "en")
sample =dataset["train"].select(range(100))

output = [{"text": x["review_body"], "stars": x["stars"]} for x in sample]

with open("data/reviews.json", "w") as f:
    json.dump(output, f, indent=2)