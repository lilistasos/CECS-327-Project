import ray
from transformers import pipeline, AutoTokenizer

@ray.remote
def analyze_sentiment(text):
    global sentiment_pipeline, tokenizer
    try:
        sentiment_pipeline
        tokenizer
    except NameError:
        sentiment_pipeline = pipeline("sentiment-analysis")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    # Truncate to 512 tokens
    tokens = tokenizer.encode(text, truncation=True, max_length=512)
    truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
    result = sentiment_pipeline(truncated_text)
    return {"text": truncated_text, "sentiment": result[0]["label"]}