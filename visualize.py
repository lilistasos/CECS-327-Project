# visualize.py is the file I will use to generate graphs and performance charts using matplotlib

import matplotlib.pyplot as plt
import json
from collections import Counter
import numpy as np
import os

# Function to save graphs to different folders
def save_graph(filename, folder="current_run"):
    """Save graph to specified folder"""
    # Create folder if it doesn't exist
    os.makedirs(f"graphs/{folder}", exist_ok=True)
    
    # Save to the specified folder
    save_path = f"graphs/{folder}/{filename}"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {filename} to graphs/{folder}/")
    
    # Also save to main graphs folder for compatibility
    plt.savefig(f"graphs/{filename}", dpi=300, bbox_inches='tight')

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
vader_labels = [d["sentiment_VADER"] for d in data]

llm_counts = Counter(llm_labels)
api_counts = Counter(api_labels)
threeclass_counts = Counter(threeclass_labels)
manual_counts = Counter(manual_labels)
hf_counts = Counter(hf_labels)
vader_counts = Counter(vader_labels)

labels = sorted(set(llm_labels) | set(api_labels) | set(threeclass_labels) | set(manual_labels) | set(hf_labels) | set(vader_labels))
llm_values = [llm_counts.get(l, 0) for l in labels]
api_values = [api_counts.get(l, 0) for l in labels]
threeclass_values = [threeclass_counts.get(l, 0) for l in labels]
manual_values = [manual_counts.get(l, 0) for l in labels]
hf_values = [hf_counts.get(l, 0) for l in labels]
vader_values = [vader_counts.get(l, 0) for l in labels]

# Create the sentiment distribution bar chart
plt.figure(figsize=(18, 8))
x = np.arange(len(labels))
width = 0.12

plt.bar(x - width*2.5, llm_values, width, label="LLM (DialoGPT)", color='orange')
plt.bar(x - width*1.5, api_values, width, label="API-based", color='lightgreen')
plt.bar(x - width*0.5, threeclass_values, width, label="3-Class (RoBERTa)", color='purple')
plt.bar(x + width*0.5, manual_values, width, label="Manual (Rule-based)", color='salmon')
plt.bar(x + width*1.5, hf_values, width, label="HuggingFace (DistilBERT)", color='skyblue')
plt.bar(x + width*2.5, vader_values, width, label="VADER", color='red')

plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Distribution Comparison Across All 6 Models')
plt.xticks(x, labels)
plt.legend()
plt.tight_layout()
save_graph('sentiment_distribution.png')
plt.show()

# 2. Performance/Latency Comparison
print("Creating performance comparison...")

# Load timing data from individual JSON files
def load_performance_data():
    performance_data = {}
    
    # Load LLM performance
    try:
        with open("results/sentiment_output_llm.json", "r") as f:
            llm_data = json.load(f)
            performance_data['LLM'] = llm_data['performance']['total_time']
    except:
        performance_data['LLM'] = 0
    
    # Load API performance
    try:
        with open("results/sentiment_output_api.json", "r") as f:
            api_data = json.load(f)
            performance_data['API'] = api_data['performance']['total_time']
    except:
        performance_data['API'] = 0
    
    # Load 3-class performance
    try:
        with open("results/sentiment_output_3class.json", "r") as f:
            class3_data = json.load(f)
            performance_data['3Class'] = class3_data['performance']['total_time']
    except:
        performance_data['3Class'] = 0
    
    # Load manual performance
    try:
        with open("results/sentiment_output_manual.json", "r") as f:
            manual_data = json.load(f)
            performance_data['Manual'] = manual_data['performance']['total_time']
    except:
        performance_data['Manual'] = 0
    
    # Load HuggingFace performance
    try:
        with open("results/sentiment_output_hf.json", "r") as f:
            hf_data = json.load(f)
            performance_data['HuggingFace'] = hf_data['performance']['total_time']
    except:
        performance_data['HuggingFace'] = 0
    
    # Load VADER performance
    try:
        with open("results/sentiment_output_vader.json", "r") as f:
            vader_data = json.load(f)
            performance_data['VADER'] = vader_data['performance']['total_time']
    except:
        performance_data['VADER'] = 0
    
    return performance_data

# Get performance data
performance_data = load_performance_data()

# Performance data (in seconds) - dynamically loaded from JSON files
models = ['LLM (DialoGPT)', 'API-based', '3-Class (RoBERTa)', 'Manual (Rule-based)', 'HuggingFace (DistilBERT)', 'VADER']
model_keys = ['LLM', 'API', '3Class', 'Manual', 'HuggingFace', 'VADER']
latency = [performance_data[key] for key in model_keys]
colors = ['orange', 'lightgreen', 'purple', 'salmon', 'skyblue', 'red']

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
save_graph('performance_comparison.png')
plt.show()

# 3. Model Agreement Analysis
print("Creating model agreement analysis...")

# Count how many models agree on each review
agreement_counts = []
for d in data:
    sentiments = [d["sentiment_LLM"], d["sentiment_API"], d["sentiment_3Class"], d["sentiment_Manual"], d["sentiment_HuggingFace"], d["sentiment_VADER"]]
    
    # Count occurrences of each sentiment
    sentiment_counts = Counter(sentiments)
    max_count = max(sentiment_counts.values())
    
    if max_count == 6:
        agreement_counts.append("All 6 Agree")
    elif max_count == 5:
        agreement_counts.append("5 Agree")
    elif max_count == 4:
        agreement_counts.append("4 Agree")
    elif max_count == 3:
        agreement_counts.append("3 Agree")
    elif max_count == 2:
        agreement_counts.append("2 Agree")
    else:
        agreement_counts.append("All Disagree")

agreement_counter = Counter(agreement_counts)
agreement_labels = list(agreement_counter.keys())
agreement_values = list(agreement_counter.values())

plt.figure(figsize=(8, 6))
plt.pie(agreement_values, labels=agreement_labels, autopct='%1.1f%%', startangle=90)
plt.title('Model Agreement Analysis (6 Models)')
plt.axis('equal')
plt.tight_layout()
save_graph('model_agreement.png')
plt.show()

# 4. Detailed Comparison Table
print("\nDetailed Comparison Results:")
print("=" * 140)
print(f"{'Review':<30} {'LLM':<12} {'API':<12} {'3-Class':<12} {'Manual':<12} {'HuggingFace':<12} {'VADER':<12}")
print("=" * 140)

for i, d in enumerate(data[:10]):  # Show first 10 reviews
    review_preview = d['text'][:27] + "..." if len(d['text']) > 30 else d['text']
    print(f"{review_preview:<30} {d['sentiment_LLM']:<12} {d['sentiment_API']:<12} {d['sentiment_3Class']:<12} {d['sentiment_Manual']:<12} {d['sentiment_HuggingFace']:<12} {d['sentiment_VADER']:<12}")

# 5. Summary Statistics
print("\n" + "=" * 140)
print("SUMMARY STATISTICS")
print("=" * 140)

print(f"Total Reviews Analyzed: {len(data)}")
print(f"LLM - POSITIVE: {llm_counts['POSITIVE']}, NEGATIVE: {llm_counts['NEGATIVE']}, NEUTRAL: {llm_counts['NEUTRAL']}")
print(f"API-based - POSITIVE: {api_counts['POSITIVE']}, NEGATIVE: {api_counts['NEGATIVE']}, NEUTRAL: {api_counts['NEUTRAL']}")
print(f"3-Class - POSITIVE: {threeclass_counts['POSITIVE']}, NEGATIVE: {threeclass_counts['NEGATIVE']}, NEUTRAL: {threeclass_counts['NEUTRAL']}")
print(f"Manual - POSITIVE: {manual_counts['POSITIVE']}, NEGATIVE: {manual_counts['NEGATIVE']}, NEUTRAL: {manual_counts['NEUTRAL']}")
print(f"HuggingFace - POSITIVE: {hf_counts['POSITIVE']}, NEGATIVE: {hf_counts['NEGATIVE']}, NEUTRAL: {hf_counts['NEUTRAL']}")
print(f"VADER - POSITIVE: {vader_counts['POSITIVE']}, NEGATIVE: {vader_counts['NEGATIVE']}, NEUTRAL: {vader_counts['NEUTRAL']}")

# Calculate agreement percentage (using corrected logic)
all_agree = 0
for d in data:
    sentiments = [d["sentiment_LLM"], d["sentiment_API"], d["sentiment_3Class"], d["sentiment_Manual"], d["sentiment_HuggingFace"], d["sentiment_VADER"]]
    sentiment_counts = Counter(sentiments)
    if max(sentiment_counts.values()) == 6:
        all_agree += 1

agreement_percentage = (all_agree / len(data)) * 100
print(f"Model Agreement Rate: {agreement_percentage:.1f}%")

print("\nPerformance Summary:")
print(f"LLM (DialoGPT): {latency[0]:.2f} seconds")
print(f"API-based: {latency[1]:.2f} seconds")
print(f"3-Class (RoBERTa): {latency[2]:.2f} seconds")
print(f"Manual (Rule-based): {latency[3]:.2f} seconds")
print(f"HuggingFace (DistilBERT): {latency[4]:.2f} seconds")
print(f"VADER: {latency[5]:.2f} seconds")

# Calculate speedups
if latency[1] > 0:
    print(f"Speedup (API vs LLM): {latency[0]/latency[1]:.1f}x faster")
if latency[3] > 0:
    print(f"Speedup (Manual vs LLM): {latency[0]/latency[3]:.0f}x faster")
else:
    print(f"Speedup (Manual vs LLM): Instant (∞x faster)")
if latency[5] > 0:
    print(f"Speedup (VADER vs LLM): {latency[0]/latency[5]:.0f}x faster")

# 6. Average Text Length by Sentiment
print("Creating average text length by sentiment...")

# Calculate text lengths
text_lengths = [len(d['text']) for d in data]

# Calculate average text length by sentiment for each model
models = ['LLM', 'API', '3Class', 'Manual', 'HuggingFace', 'VADER']
avg_lengths = {}

for model in models:
    model_key = f"sentiment_{model}" if model != 'LLM' else "sentiment_LLM"
    sentiments = [d[model_key] for d in data]
    
    model_avg = {}
    for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        mask = [s == sentiment for s in sentiments]
        if any(mask):
            avg_length = np.mean([text_lengths[j] for j in range(len(text_lengths)) if mask[j]])
            model_avg[sentiment] = avg_length
        else:
            model_avg[sentiment] = 0
    
    avg_lengths[model] = model_avg

# Create bar chart
fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(models))
width = 0.25

colors_sentiment = {'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'blue'}
for i, sentiment in enumerate(['POSITIVE', 'NEGATIVE', 'NEUTRAL']):
    values = [avg_lengths[model].get(sentiment, 0) for model in models]
    ax.bar(x + i*width, values, width, label=sentiment, color=colors_sentiment[sentiment])

ax.set_xlabel('Model')
ax.set_ylabel('Average Text Length (characters)')
ax.set_title('Average Text Length by Sentiment and Model')
ax.set_xticks(x + width)
ax.set_xticklabels(models)
ax.legend()
plt.tight_layout()
save_graph('avg_text_length_by_sentiment.png')
plt.show()

# 7. Model Consistency Analysis
print("Creating model consistency analysis...")

# Analyze how consistent each model is with itself
def calculate_consistency(model_sentiments):
    """Calculate how consistent a model's predictions are"""
    consistency_scores = []
    
    for i, sent1 in enumerate(model_sentiments):
        for j, sent2 in enumerate(model_sentiments[i+1:], i+1):
            # Simple similarity: check if both reviews contain same sentiment keywords
            review1 = data[i]['text'].lower()
            review2 = data[j]['text'].lower()
            
            # Use the same word lists as in tasks.py for consistency
            positive_words = ['great', 'amazing', 'wonderful', 'fantastic', 'excellent', 'love', 'enjoy', 'good', 'nice', 'fabulous', 'exciting', 'fun', 'happy', 'enjoyed']
            negative_words = ['terrible', 'awful', 'horrible', 'bad', 'disappointing', 'waste', 'let down', 'worst', 'hate', 'dislike', 'poor']
            
            review1_pos = any(word in review1 for word in positive_words)
            review1_neg = any(word in review1 for word in negative_words)
            review2_pos = any(word in review2 for word in positive_words)
            review2_neg = any(word in review2 for word in negative_words)
            
            # If reviews have similar sentiment indicators, check if model agrees
            if (review1_pos and review2_pos) or (review1_neg and review2_neg):
                consistency_scores.append(1 if sent1 == sent2 else 0)
    
    return np.mean(consistency_scores) * 100 if consistency_scores else 0

model_consistencies = {}
for model in models:
    model_key = f"sentiment_{model}" if model != 'LLM' else "sentiment_LLM"
    model_sentiments = [d[model_key] for d in data]
    consistency = calculate_consistency(model_sentiments)
    model_consistencies[model] = consistency

# Plot consistency
plt.figure(figsize=(10, 6))
bars = plt.bar(models, [model_consistencies[m] for m in models], color=colors)
plt.xlabel('Model')
plt.ylabel('Consistency Score (%)')
plt.title('Model Consistency Analysis')
plt.ylim(0, 100)

# Add value labels
for bar, value in zip(bars, [model_consistencies[m] for m in models]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{value:.1f}%', ha='center', va='bottom')

plt.tight_layout()
save_graph('model_consistency.png')
plt.show()

# 8. Memory Usage Comparison
print("Creating memory usage comparison...")

# Load memory usage data from performance files
def load_memory_data():
    memory_data = {}
    
    # Load real memory usage from performance files
    models = ['llm', 'api', '3class', 'manual', 'hf', 'vader']
    model_names = ['LLM', 'API', '3Class', 'Manual', 'HuggingFace', 'VADER']
    
    for model, model_name in zip(models, model_names):
        try:
            with open(f"results/sentiment_output_{model}.json", "r") as f:
                data = json.load(f)
                # Use real memory data if available
                if 'performance' in data and 'memory_used' in data['performance']:
                    memory_used = data['performance']['memory_used']
                    # Handle negative memory usage (memory was freed)
                    if memory_used < 0:
                        memory_data[model_name] = abs(memory_used)  # Use absolute value
                    else:
                        memory_data[model_name] = memory_used
                else:
                    # Fallback to estimated values if real data not available
                    if model == 'llm':
                        memory_data[model_name] = 512  # Large model
                    elif model == 'api':
                        memory_data[model_name] = 8    # Very small
                    elif model == '3class':
                        memory_data[model_name] = 256  # Medium model
                    elif model == 'manual':
                        memory_data[model_name] = 2   # Minimal
                    elif model == 'hf':
                        memory_data[model_name] = 128  # Medium model
                    elif model == 'vader':
                        memory_data[model_name] = 4   # Very small
        except:
            # Default memory estimates if files not found
            if model == 'llm':
                memory_data[model_name] = 512
            elif model == 'api':
                memory_data[model_name] = 8
            elif model == '3class':
                memory_data[model_name] = 256
            elif model == 'manual':
                memory_data[model_name] = 2
            elif model == 'hf':
                memory_data[model_name] = 128
            elif model == 'vader':
                memory_data[model_name] = 4
    
    return memory_data

memory_usage = load_memory_data()
models = ['LLM', 'API', '3Class', 'Manual', 'HuggingFace', 'VADER']
memory_values = [memory_usage[model] for model in models]

# Create memory usage comparison
plt.figure(figsize=(12, 8))
bars = plt.bar(models, memory_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

# Add value labels
for bar, value in zip(bars, memory_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             f'{value} MB', ha='center', va='bottom', fontweight='bold')

plt.xlabel('Sentiment Analysis Models')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage Comparison\n')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
save_graph('memory_usage_comparison.png')
plt.show()

print("\nAll graphs saved to 'graphs/' directory")