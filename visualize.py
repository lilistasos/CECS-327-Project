# visualize.py is the file I will use to generate graphs and performance charts using matplotlib

import matplotlib.pyplot as plt
import json
from collections import Counter
import numpy as np
import os

# Global constants to reduce redundancy
MODELS = ['LLM', 'API', '3Class', 'Manual', 'HuggingFace', 'VADER']
MODEL_FILES = ['llm', 'api', '3class', 'manual', 'hf', 'vader']
MODEL_DISPLAY_NAMES = ['LLM (DialoGPT)', 'API-based', '3-Class (RoBERTa)', 'Manual (Rule-based)', 'HuggingFace (DistilBERT)', 'VADER']
COLORS = ['orange', 'lightgreen', 'purple', 'salmon', 'skyblue', 'red']

# Import word lists from tasks.py to avoid duplication
from tasks import POSITIVE_WORDS, NEGATIVE_WORDS

# Function to save graphs to different folders
def save_graph(filename, folder="current_run"):
    """Save graph to specified folder only"""
    # Create folder if it doesn't exist
    os.makedirs(f"graphs/{folder}", exist_ok=True)
    
    # Save only to the specified folder
    save_path = f"graphs/{folder}/{filename}"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {filename} to graphs/{folder}/")

# Function to run visualization with specified folder
def run_visualization(folder="current_run"):
    """Run all visualizations and save to specified folder"""
    
    # Load comparison results
    with open("results/sentiment_output_comparison.json") as f:
        data = json.load(f)

    # 1. Sentiment Distribution Comparison
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
    save_graph('sentiment_distribution.png', folder)
    plt.show()

    # 2. Performance/Latency Comparison
    # Generic function to load data from JSON files
    def load_json_data(data_type):
        """Load performance or memory data from JSON files"""
        data = {}
        for model, model_name in zip(MODEL_FILES, MODELS):
            try:
                with open(f"results/sentiment_output_{model}.json", "r") as f:
                    json_data = json.load(f)
                    if 'performance' in json_data and data_type in json_data['performance']:
                        value = json_data['performance'][data_type]
                        # Handle negative memory usage (memory was freed)
                        if data_type == 'memory_used' and value < 0:
                            data[model_name] = abs(value)
                        else:
                            data[model_name] = value
            except:
                # Set default value for performance data, skip for memory data
                if data_type == 'total_time':
                    data[model_name] = 0
        return data

    # Get performance data
    performance_data = load_json_data('total_time')

    # Performance data (in seconds) - dynamically loaded from JSON files
    latency = [performance_data[model] for model in MODELS]

    plt.figure(figsize=(18, 10))
    bars = plt.bar(MODEL_DISPLAY_NAMES, latency, color=COLORS)
    plt.xlabel('Model Type')
    plt.ylabel('Latency (seconds)')
    plt.title('Performance Comparison: Latency vs Model Type')
    plt.yscale('log')  # Use log scale to show the huge difference

    # Add value labels on bars
    for bar, value in zip(bars, latency):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2f}s', ha='center', va='bottom', fontweight='bold')
        plt.text(bar.get_x() + bar.get_width()/2, 0.01,
                f'{value:.2f}s', ha='center', va='bottom', fontweight='bold')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.25)  # Add more space at bottom for labels
    save_graph('performance_comparison.png', folder)
    plt.show()

    # 3. Model Agreement Analysis
    # Count how many models agree on each review
    agreement_counts = []
    for d in data:
        sentiments = [d[f"sentiment_{model}" if model != 'LLM' else "sentiment_LLM"] for model in MODELS]
        
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
    save_graph('model_agreement.png', folder)
    plt.show()

    # 6. Average Text Length by Sentiment
    # Calculate text lengths
    text_lengths = [len(d['text']) for d in data]

    # Calculate average text length by sentiment for each model
    avg_lengths = {}

    for model in MODELS:
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

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(MODELS))
    width = 0.25

    positive_vals = [avg_lengths[model]['POSITIVE'] for model in MODELS]
    negative_vals = [avg_lengths[model]['NEGATIVE'] for model in MODELS]
    neutral_vals = [avg_lengths[model]['NEUTRAL'] for model in MODELS]

    ax.bar(x - width, positive_vals, width, label='Positive', color='green', alpha=0.7)
    ax.bar(x, negative_vals, width, label='Negative', color='red', alpha=0.7)
    ax.bar(x + width, neutral_vals, width, label='Neutral', color='gray', alpha=0.7)

    ax.set_xlabel('Models')
    ax.set_ylabel('Average Text Length (characters)')
    ax.set_title('Average Text Length by Sentiment for Each Model')
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=45)
    ax.legend()
    plt.tight_layout()
    save_graph('avg_text_length_by_sentiment.png', folder)
    plt.show()

    # 7. Model Consistency Analysis
    def calculate_consistency(model_sentiments):
        """Calculate consistency score for a model"""
        consistency_scores = []
        
        for i in range(len(data)):
            for j in range(i+1, min(i+10, len(data))):  # Compare with next 10 reviews
                review1 = data[i]['text'].lower()
                review2 = data[j]['text'].lower()
                
                # Use imported word lists from tasks.py for consistency
                
                review1_pos = any(word in review1 for word in POSITIVE_WORDS)
                review1_neg = any(word in review1 for word in NEGATIVE_WORDS)
                review2_pos = any(word in review2 for word in POSITIVE_WORDS)
                review2_neg = any(word in review2 for word in NEGATIVE_WORDS)
                
                # If reviews have similar sentiment indicators, check if model agrees
                if (review1_pos and review2_pos) or (review1_neg and review2_neg):
                    consistency_scores.append(1 if model_sentiments[i] == model_sentiments[j] else 0)
        
        return np.mean(consistency_scores) * 100 if consistency_scores else 0

    model_consistencies = {}
    for model in MODELS:
        model_key = f"sentiment_{model}" if model != 'LLM' else "sentiment_LLM"
        model_sentiments = [d[model_key] for d in data]
        consistency = calculate_consistency(model_sentiments)
        model_consistencies[model] = consistency

    # Plot consistency
    plt.figure(figsize=(10, 6))
    bars = plt.bar(MODELS, [model_consistencies[m] for m in MODELS], color=COLORS)
    plt.xlabel('Model')
    plt.ylabel('Consistency Score (%)')
    plt.title('Model Consistency Analysis')
    plt.ylim(0, 100)

    # Add value labels
    for bar, value in zip(bars, [model_consistencies[m] for m in MODELS]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    save_graph('model_consistency.png', folder)
    plt.show()

    # 8. Memory Usage Comparison
    # Load memory usage data
    memory_usage = load_json_data('memory_used')
    
    # Only create memory usage graph if we have real data
    if memory_usage:
        # Get only models with real memory data
        available_models = list(memory_usage.keys())
        memory_values = [memory_usage[model] for model in available_models]
        available_colors = ['orange', 'lightgreen', 'purple', 'salmon', 'skyblue', 'red'][:len(available_models)]

        # Create memory usage comparison
        plt.figure(figsize=(12, 8))
        bars = plt.bar(available_models, memory_values, color=available_colors, alpha=0.8, edgecolor='black', linewidth=1)

        # Calculate proper y-axis limits
        max_value = max(memory_values) if memory_values else 0
        y_limit = max_value * 1.2  # Add 20% padding for labels
        
        # Add value labels on bars
        for bar, value in zip(bars, memory_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (y_limit * 0.02),
                    f'{value:.1f} MB', ha='center', va='bottom', fontweight='bold')

        plt.xlabel('Sentiment Analysis Models')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage Comparison\n(Only models with real memory data shown)')
        plt.xticks(rotation=45)
        plt.ylim(0, y_limit)  # Set proper y-axis limits
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        save_graph('memory_usage_comparison.png', folder)
        plt.show()
    # If no real memory data available, skip this graph

    # All graphs saved to specified folder

# Run visualization with default folder
if __name__ == "__main__":
    run_visualization()