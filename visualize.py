# visualize.py is the file I will use to generate graphs and performance charts using matplotlib

import matplotlib as plt
import json
import matplotlib.pyplot as plt
from collections import Counter

# Load comparison results
with open("results/sentiment_output_comparison.json") as f:
    data = json.load(f)

# Count sentiment labels for each model
hf_labels = [d["sentiment_HuggingFace"] for d in data]
gpt_labels = [d["sentiment_OpenAI"] for d in data]

hf_counts = Counter(hf_labels)
gpt_counts = Counter(gpt_labels)

labels = sorted(set(hf_labels) | set(gpt_labels))
hf_values = [hf_counts.get(l, 0) for l in labels]
gpt_values = [gpt_counts.get(l, 0) for l in labels]

x = range(len(labels))
plt.figure(figsize=(8, 5))
plt.bar(x, hf_values, width=0.4, label="HuggingFace", align="center")
plt.bar([i + 0.4 for i in x], gpt_values, width=0.4, label="OpenAI", align="center")
plt.xticks([i + 0.2 for i in x], labels)
plt.ylabel("Count")
plt.title("Sentiment Label Distribution: HuggingFace vs OpenAI")
plt.legend()
plt.tight_layout()
plt.show()

# Print a few example reviews with both sentiments
print("\nSample comparison:")
for d in data[:5]:
    print(f"Review: {d['text'][:100]}...")
    print(f"  HuggingFace: {d['sentiment_HuggingFace']}")
    print(f"  OpenAI:      {d['sentiment_OpenAI']}")
    print()

