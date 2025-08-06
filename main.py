# libraries needed for this project:

import ray
from tasks import analyze_sentiment, analyze_sentiment_api_simple, analyze_sentiment_llm_safe, analyze_sentiment_3class, analyze_sentiment_manual, analyze_sentiment_vader
import pandas as pd
import json
import os
import time
import psutil
import gc

# Starting section of the program, where the user is asked to choose between a single run or three runs.
# Single run is used for testing purposes, while three runs is used for averaging the results.
def get_user_choice():
    print("\n" + "="*60)
    print("DISTRIBUTED SENTIMENT ANALYSIS SYSTEM")
    print("="*60)
    print("\nRun Options:")
    print("1. Single Run (saves to current_run)")
    print("2. Three Runs (saves to average_run)")
    
    while True:
        try:
            choice = input("\nSelect option (1-2): ").strip()
            if choice in ['1', '2']:
                return int(choice)
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()
        except:
            print("Invalid input. Please try again.")

# This function runs the complete sentiment analysis pipeline.
def run_single_analysis():
    # Shutdown Ray if it's already running
    try:
        ray.shutdown()
    except:
        pass
    
    # Initialize Ray (used for parallel processing)
    ray.init()

    # Memory monitoring functions
    # This function gets the current memory usage in MB.
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    # This function prints the current memory usage in MB.
    def print_memory_status(stage=""):
        memory_mb = get_memory_usage()
        print(f"Memory usage {stage}: {memory_mb:.1f} MB")
        return memory_mb

    # Read the CSV file
    df = pd.read_csv("data/DisneylandReviews.csv", encoding="latin1").head(1000)
    reviews = df['Review_Text'].dropna().tolist()

    # Print the dataset size and number of reviews to process.
    print(f"Dataset size: {len(df)} reviews")
    print(f"Reviews to process: {len(reviews)} reviews")
    print_memory_status("after loading data")

    # Run LLM sentiment analysis
    print("\nRunning LLM sentiment analysis...")
    llm_memory_before = print_memory_status("before LLM")
    start = time.time()
    futures_llm = [analyze_sentiment_llm_safe.remote(r) for r in reviews]
    results_llm = ray.get(futures_llm)
    llm_time = time.time() - start
    print(f"LLM time: {llm_time:.2f} seconds")
    llm_memory_after = print_memory_status("after LLM")
    gc.collect() 

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

    # Sample comparisons
    print("\nSample comparisons:")
    for i, d in enumerate(comparison_results[:5]):
        print(f"Review {i+1}: {d['text'][:50]}...")
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
    print("- 3-class Hugging Face sentiment analysis (RoBERta)")
    print("- Manual sentiment analysis (Rule-based)")
    print("- Hugging Face sentiment analysis (DistilBERT)")
    print("- VADER sentiment analysis (Valence Aware Dictionary)")
    print("- Results saved!")

    return True

# This function cleans the main graphs folder.
def clean_main_graphs_folder():
    source_dir = "graphs"
    for file in os.listdir(source_dir):
        if file.endswith('.png'):
            file_path = os.path.join(source_dir, file)
            os.remove(file_path)
            print(f"Removed {file} from main graphs folder")

# Get user choice and print latency to console.
choice = get_user_choice()

if choice == 1:
    # Single run
    print("\nRunning single analysis...")
    start_time = time.time()
    if run_single_analysis():
        total_time = time.time() - start_time
        print(f"\nSingle analysis completed successfully!")
        print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print("Running visualization...")
        import visualize
        visualize.run_visualization("current_run")
        clean_main_graphs_folder()
        print("Results saved to graphs/current_run/")
    
elif choice == 2:
    # Three runs
    print("\nRunning three analyses for averaging...")
    start_time = time.time()
    successful_runs = 0
    
    for i in range(3):
        print(f"\n--- Run {i+1}/3 ---")
        run_start = time.time()
        if run_single_analysis():
            run_time = time.time() - run_start
            successful_runs += 1
            print(f"Run {i+1} completed successfully in {run_time:.2f} seconds")
        else:
            print(f"Run {i+1} failed")
    
    total_time = time.time() - start_time
    if successful_runs > 0:
        print(f"\nCompleted {successful_runs} successful runs")
        print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Average time per run: {total_time/successful_runs:.2f} seconds")
        print("Running visualization...")
        import visualize
        visualize.run_visualization("average_run")
        clean_main_graphs_folder()
        print("Results saved to graphs/average_run/")
    else:
        print("No successful runs completed")