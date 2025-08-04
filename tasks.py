import ray
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Global word lists for sentiment analysis
POSITIVE_WORDS = ['great', 'amazing', 'wonderful', 'fantastic', 'excellent', 'love', 'enjoy', 'good', 'nice', 'fabulous', 'exciting', 'fun', 'happy', 'enjoyed', 'perfect']
NEGATIVE_WORDS = ['terrible', 'awful', 'horrible', 'bad', 'disappointing', 'waste', 'let down', 'worst', 'hate', 'dislike', 'poor']

# Global word lists for LLM response parsing
LLM_POSITIVE_INDICATORS = ['positive', 'good', 'great', 'excellent', 'amazing', 'wonderful']
LLM_NEGATIVE_INDICATORS = ['negative', 'bad', 'terrible', 'awful', 'horrible', 'poor']
LLM_NEUTRAL_INDICATORS = ['neutral', 'okay', 'fine', 'average', 'normal']

# Global model variables (will be initialized on first use)
sentiment_pipeline = None
distilbert_tokenizer = None
llm_tokenizer = None
llm_model = None
sentiment_3class_pipeline = None
vader_analyzer = None


# Rule-based sentiment analysis
def _count_words(text, word_list):
    """Helper function to count words in text"""
    text_lower = text.lower()
    return sum(1 for word in word_list if word in text_lower)

def _rule_based_sentiment(text):
    """Helper function for rule-based sentiment analysis"""
    pos_count = _count_words(text, POSITIVE_WORDS)
    neg_count = _count_words(text, NEGATIVE_WORDS)
    
    if pos_count > neg_count:
        return "POSITIVE"
    elif neg_count > pos_count:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

# This function uses Hugging Face pipeline for sentiment analysis.
@ray.remote
def analyze_sentiment(text):
    global sentiment_pipeline, distilbert_tokenizer
    if sentiment_pipeline is None:
        sentiment_pipeline = pipeline("sentiment-analysis")
        distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    
    tokens = distilbert_tokenizer.encode(text, truncation=True, max_length=512)
    truncated_text = distilbert_tokenizer.decode(tokens, skip_special_tokens=True)
    result = sentiment_pipeline(truncated_text)
    return {"text": truncated_text, "sentiment": result[0]["label"]}


# API-based sentiment analysis
@ray.remote
def analyze_sentiment_api_simple(text):
    
    # Use the helper function for sentiment analysis
    sentiment = _rule_based_sentiment(text)

    

    
    return {"text": text, "sentiment": sentiment}

# VADER sentiment analysis
@ray.remote
def analyze_sentiment_vader(text):
    global vader_analyzer
    if vader_analyzer is None:
        vader_analyzer = SentimentIntensityAnalyzer()
    
    # Get VADER sentiment scores
    scores = vader_analyzer.polarity_scores(text)
    
    # VADER provides compound score between -1 and 1
    compound_score = scores['compound']
    
    # Map VADER compound score to our sentiment categories
    if compound_score >= 0.05:
        sentiment = "POSITIVE"
    elif compound_score <= -0.05:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    
    return {"text": text, "sentiment": sentiment}

# Manual rule-based sentiment analysis
@ray.remote
def analyze_sentiment_manual(text):
    text_lower = text.lower()
    
    # Use global word lists
    positive_count = sum(1 for word in POSITIVE_WORDS if word in text_lower)
    negative_count = sum(1 for word in NEGATIVE_WORDS if word in text_lower)
    
    if positive_count > negative_count:
        sentiment = "POSITIVE"
    elif negative_count > positive_count:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    
    return {"text": text, "sentiment": sentiment}

# LLM sentiment analysis
@ray.remote
def analyze_sentiment_llm_safe(text):
    global llm_tokenizer, llm_model
    if llm_tokenizer is None or llm_model is None:
        # Use a smaller, memory-safe LLM
        model_name = "microsoft/DialoGPT-small" 
        llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        llm_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token if not present
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
    
    # Create a clear numerical prompt explaining what each number means
    prompt = f"Review: {text}\nRate the sentiment: 1=positive, -1=negative, 0=neutral\nAnswer:"
    
    try:
        # Tokenize with truncation to prevent memory issues
        inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        
        # Generate with parameters optimized for instruction following
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=5,  
                temperature=0.1, 
                do_sample=True,    
                top_p=0.9,        
                pad_token_id=llm_tokenizer.eos_token_id,
                eos_token_id=llm_tokenizer.eos_token_id
            )
        
        # Decode the response
        response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated part (after the prompt)
        generated_text = response[len(prompt):].strip().lower()
        
        # Parse sentiment from LLM response - more flexible parsing
        generated_text = generated_text.strip()
        generated_text = generated_text.lower()
        
        # Parse numerical sentiment from LLM response
        generated_text = generated_text.strip()
        
        # Look for numerical values in the response
        if "1" in generated_text or "positive" in generated_text.lower():
            sentiment = "POSITIVE"
        elif "-1" in generated_text or "negative" in generated_text.lower():
            sentiment = "NEGATIVE"
        elif "0" in generated_text or "neutral" in generated_text.lower():
            sentiment = "NEUTRAL"
        else:
            # If LLM fails, use rule-based fallback
            sentiment = _rule_based_sentiment(text)
            
    except Exception as e:
        # If LLM completely fails, use rule-based approach
        sentiment = _rule_based_sentiment(text)
    
    return {"text": text, "sentiment": sentiment}

@ray.remote
def analyze_sentiment_3class(text):
    """
    Use a 3-class sentiment analysis model through Hugging Face.
    This will return POSITIVE, NEGATIVE, or NEUTRAL.
    """
    global sentiment_3class_pipeline
    if sentiment_3class_pipeline is None:
        # Use a simpler 3-class model that's compatible
        try:
            sentiment_3class_pipeline = pipeline("sentiment-analysis", 
                                               model="nlptown/bert-base-multilingual-uncased-sentiment")
        except:
            # Fallback to a basic 3-class approach
            sentiment_3class_pipeline = None
    
    # Truncate to prevent memory issues
    if len(text) > 500:
        text = text[:500] + "..."
    
    try:
        if sentiment_3class_pipeline is not None:
            result = sentiment_3class_pipeline(text)
            label = result[0]['label']
            
            # Map the labels to our format
            if '1' in label or '2' in label:
                sentiment = "NEGATIVE"
            elif '3' in label:
                sentiment = "NEUTRAL"
            elif '4' in label or '5' in label:
                sentiment = "POSITIVE"
            else:
                sentiment = "NEUTRAL"
        else:
            # Fallback to keyword-based 3-class analysis
            text_lower = text.lower()
            
            # Count different types of sentiment words using global lists
            very_pos_count = sum(1 for word in VERY_POSITIVE_WORDS if word in text_lower)
            pos_count = sum(1 for word in POSITIVE_WORDS if word in text_lower)
            neg_count = sum(1 for word in NEGATIVE_WORDS if word in text_lower)
            very_neg_count = sum(1 for word in VERY_NEGATIVE_WORDS if word in text_lower)
            
            # Calculate sentiment score
            positive_score = very_pos_count * 2 + pos_count
            negative_score = very_neg_count * 2 + neg_count
            
            # Determine sentiment based on scores
            if positive_score > negative_score and positive_score > 1:
                sentiment = "POSITIVE"
            elif negative_score > positive_score and negative_score > 1:
                sentiment = "NEGATIVE"
            else:
                sentiment = "NEUTRAL"
            
    except Exception as e:
        print(f"3-class sentiment analysis failed: {e}")
        # Fallback to neutral
        sentiment = "NEUTRAL"
    
    return {"text": text, "sentiment": sentiment}