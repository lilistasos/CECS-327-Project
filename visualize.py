# visualize.py is the file I will use to generate graphs and performance charts using matplotlib

import matplotlib.pyplot as plt
import json
from collections import Counter
import numpy as np

# Load comparison results
with open("results/sentiment_output_comparison.json") as f:
    data = json.load(f)

# 1. Sentiment Distribution Comparison
print("Creating sentiment distribution comparison...")

# Count sentiment labels for each model
llm_labels = [d["sentiment_LLM"] for d in data]
api_labels = [d["sentiment_API"] for d in data]
threeclass_labels = [d["sentiment_3Class"] for d in data]
manual_labels = [d["sentiment_Manual"] for d in data]
hf_labels = [d["sentiment_HuggingFace"] for d in data]

llm_counts = Counter(llm_labels)
api_counts = Counter(api_labels)
threeclass_counts = Counter(threeclass_labels)
manual_counts = Counter(manual_labels)
hf_counts = Counter(hf_labels)

labels = sorted(set(llm_labels) | set(api_labels) | set(threeclass_labels) | set(manual_labels) | set(hf_labels))
llm_values = [llm_counts.get(l, 0) for l in labels]
api_values = [api_counts.get(l, 0) for l in labels]
threeclass_values = [threeclass_counts.get(l, 0) for l in labels]
manual_values = [manual_counts.get(l, 0) for l in labels]
hf_values = [hf_counts.get(l, 0) for l in labels]

# Create the sentiment distribution bar chart
plt.figure(figsize=(16, 8))
x = np.arange(len(labels))
width = 0.15

plt.bar(x - width*2, llm_values, width, label="LLM (DialoGPT)", color='orange')
plt.bar(x - width*1, api_values, width, label="API-based", color='lightgreen')
plt.bar(x, threeclass_values, width, label="3-Class (RoBERTa)", color='purple')
plt.bar(x + width*1, manual_values, width, label="Manual (Rule-based)", color='salmon')
plt.bar(x + width*2, hf_values, width, label="HuggingFace (DistilBERT)", color='skyblue')

plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Distribution Comparison Across All 5 Models')
plt.xticks(x, labels)
plt.legend()
plt.tight_layout()
plt.savefig('graphs/sentiment_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Performance/Latency Comparison
print("Creating performance comparison...")

# Performance data (in seconds) - updated with real results from 100 reviews
models = ['LLM (DialoGPT)', 'API-based', '3-Class (RoBERTa)', 'Manual (Rule-based)', 'HuggingFace (DistilBERT)']
latency = [36.19, 0.39, 24.57, 0.00, 6.31]  # Real results from 5-model run
colors = ['orange', 'lightgreen', 'purple', 'salmon', 'skyblue']

plt.figure(figsize=(14, 8))
bars = plt.bar(models, latency, color=colors)
plt.xlabel('Model Type')
plt.ylabel('Latency (seconds)')
plt.title('Performance Comparison: Latency vs Model Type')
plt.yscale('log')  # Use log scale to show the huge difference

# Add value labels on bars
for bar, value in zip(bars, latency):
    if value > 0:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{value:.2f}s', ha='center', va='bottom')
    else:
        plt.text(bar.get_x() + bar.get_width()/2, 0.01, 
                 'Instant', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('graphs/performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Model Agreement Analysis
print("Creating model agreement analysis...")

# Count how many models agree on each review
agreement_counts = []
for d in data:
    sentiments = [d["sentiment_LLM"], d["sentiment_API"], d["sentiment_3Class"], d["sentiment_Manual"], d["sentiment_HuggingFace"]]
    unique_sentiments = len(set(sentiments))
    if unique_sentiments == 1:
        agreement_counts.append("All Agree")
    elif unique_sentiments == 2:
        agreement_counts.append("2 Agree")
    elif unique_sentiments == 3:
        agreement_counts.append("3 Agree")
    elif unique_sentiments == 4:
        agreement_counts.append("4 Agree")
    else:
        agreement_counts.append("All Disagree")

agreement_counter = Counter(agreement_counts)
agreement_labels = list(agreement_counter.keys())
agreement_values = list(agreement_counter.values())

plt.figure(figsize=(8, 6))
plt.pie(agreement_values, labels=agreement_labels, autopct='%1.1f%%', startangle=90)
plt.title('Model Agreement Analysis (5 Models)')
plt.axis('equal')
plt.tight_layout()
plt.savefig('graphs/model_agreement.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Detailed Comparison Table
print("\nDetailed Comparison Results:")
print("=" * 120)
print(f"{'Review':<30} {'LLM':<12} {'API':<12} {'3-Class':<12} {'Manual':<12} {'HuggingFace':<12}")
print("=" * 120)

for i, d in enumerate(data[:10]):  # Show first 10 reviews
    review_preview = d['text'][:27] + "..." if len(d['text']) > 30 else d['text']
    print(f"{review_preview:<30} {d['sentiment_LLM']:<12} {d['sentiment_API']:<12} {d['sentiment_3Class']:<12} {d['sentiment_Manual']:<12} {d['sentiment_HuggingFace']:<12}")

# 5. Summary Statistics
print("\n" + "=" * 120)
print("SUMMARY STATISTICS")
print("=" * 120)

print(f"Total Reviews Analyzed: {len(data)}")
print(f"LLM - POSITIVE: {llm_counts['POSITIVE']}, NEGATIVE: {llm_counts['NEGATIVE']}, NEUTRAL: {llm_counts['NEUTRAL']}")
print(f"API-based - POSITIVE: {api_counts['POSITIVE']}, NEGATIVE: {api_counts['NEGATIVE']}, NEUTRAL: {api_counts['NEUTRAL']}")
print(f"3-Class - POSITIVE: {threeclass_counts['POSITIVE']}, NEGATIVE: {threeclass_counts['NEGATIVE']}, NEUTRAL: {threeclass_counts['NEUTRAL']}")
print(f"Manual - POSITIVE: {manual_counts['POSITIVE']}, NEGATIVE: {manual_counts['NEGATIVE']}, NEUTRAL: {manual_counts['NEUTRAL']}")
print(f"HuggingFace - POSITIVE: {hf_counts['POSITIVE']}, NEGATIVE: {hf_counts['NEGATIVE']}, NEUTRAL: {hf_counts['NEUTRAL']}")

# Calculate agreement percentage
all_agree = sum(1 for d in data if len(set([d["sentiment_LLM"], d["sentiment_API"], d["sentiment_3Class"], d["sentiment_Manual"], d["sentiment_HuggingFace"]])) == 1)
agreement_percentage = (all_agree / len(data)) * 100
print(f"Model Agreement Rate: {agreement_percentage:.1f}%")

print("\nPerformance Summary:")
print(f"LLM (DialoGPT): {latency[0]:.2f} seconds")
print(f"API-based: {latency[1]:.2f} seconds")
print(f"3-Class (RoBERTa): {latency[2]:.2f} seconds")
print(f"Manual (Rule-based): {latency[3]:.2f} seconds")
print(f"HuggingFace (DistilBERT): {latency[4]:.2f} seconds")
print(f"Speedup (API vs LLM): {latency[0]/latency[1]:.1f}x faster")
if latency[3] > 0:
    print(f"Speedup (Manual vs LLM): {latency[0]/latency[3]:.0f}x faster")
else:
    print(f"Speedup (Manual vs LLM): Instant (∞x faster)")

print("\n✅ All graphs saved to 'graphs/' directory!")
print("📊 Generated visualizations:")
print("   - sentiment_distribution.png")
print("   - performance_comparison.png") 
print("   - model_agreement.png")

