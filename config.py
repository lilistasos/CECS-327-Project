import json

reviews = []
with open("data/yelp_academic_dataset_review.json") as f:
    for line in f:
        reviews.append(json.loads(line))
