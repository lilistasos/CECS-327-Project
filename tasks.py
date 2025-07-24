import ray
from transformers import pipeline, AutoTokenizer
import openai
from dotenv import load_dotenv
load_dotenv()
import os

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

@ray.remote
def analyze_sentiment_chatgpt(text):
    from dotenv import load_dotenv
    load_dotenv()
    import os
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"Classify the sentiment of this review as Positive, Negative, or Neutral:\n\n{text}"
    try:
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        sentiment = response.choices[0].message.content.strip()
        return {"text": text, "sentiment": sentiment}
    except Exception as e:
        print(f"OpenAI API error for text: {text[:50]}... Error: {e}")
        return {"text": text, "sentiment": "ERROR"}