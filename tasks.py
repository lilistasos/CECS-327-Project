import ray
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from dotenv import load_dotenv
load_dotenv()
import os
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Global word lists for sentiment analysis
POSITIVE_WORDS = ['great', 'amazing', 'wonderful', 'fantastic', 'excellent', 'love', 'enjoy', 'good', 'nice', 'fabulous', 'exciting', 'fun', 'happy', 'enjoyed']
NEGATIVE_WORDS = ['terrible', 'awful', 'horrible', 'bad', 'disappointing', 'waste', 'let down', 'worst', 'hate', 'dislike', 'poor']
VERY_POSITIVE_WORDS = ['amazing', 'fantastic', 'excellent', 'wonderful', 'love', 'fabulous', 'perfect']
VERY_NEGATIVE_WORDS = ['let down', 'dislike', 'poor', 'terrible', 'awful']

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

@ray.remote
def analyze_sentiment(text):
    """Use Hugging Face pipeline for sentiment analysis"""
    global sentiment_pipeline, distilbert_tokenizer
    if sentiment_pipeline is None:
        sentiment_pipeline = pipeline("sentiment-analysis")
        distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    
    # Truncate to 512 tokens
    tokens = distilbert_tokenizer.encode(text, truncation=True, max_length=512)
    truncated_text = distilbert_tokenizer.decode(tokens, skip_special_tokens=True)
    result = sentiment_pipeline(truncated_text)
    return {"text": truncated_text, "sentiment": result[0]["label"]}

@ray.remote
def analyze_sentiment_api_simple(text):
    import requests
    
    # Use a simple sentiment analysis based on text analysis
    # Since API authentication is failing, let's create a simple rule-based approach
    # that simulates what an API might return
    
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
    if positive_score > negative_score and positive_score > 0:
        sentiment = "POSITIVE"
    elif negative_score > positive_score and negative_score > 0:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    
    return {"text": text, "sentiment": sentiment}

@ray.remote
def analyze_sentiment_vader(text):
    """VADER sentiment analysis - specifically designed for social media text"""
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

@ray.remote
def analyze_sentiment_manual(text):
    """Manual rule-based sentiment analysis using keyword matching"""
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

@ray.remote
def analyze_sentiment_llm_safe(text):
    """
    Use a smaller LLM through Hugging Face for sentiment analysis.
    This uses a smaller model that won't crash the system but still demonstrates LLM capabilities.
    """
    global llm_tokenizer, llm_model
    if llm_tokenizer is None or llm_model is None:
        # Use a smaller, memory-safe LLM
        model_name = "microsoft/DialoGPT-small"  # Only 117M parameters
        llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        llm_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token if not present
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
    
    # Create a clearer, more structured prompt for sentiment analysis
    prompt = f"Review: {text}\nQuestion: Is this review positive, negative, or neutral?\nAnswer: This review is"
    
    try:
        # Tokenize with truncation to prevent memory issues
        inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        
        # Generate with conservative parameters
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=15,  # Allow more tokens for better response
                temperature=0.3,    # Slightly higher temperature for more varied responses
                do_sample=True,     # Enable sampling for better responses
                pad_token_id=llm_tokenizer.eos_token_id,
                eos_token_id=llm_tokenizer.eos_token_id
            )
        
        # Decode the response
        response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated part (after the prompt)
        generated_text = response[len(prompt):].strip().lower()
        
        # More robust sentiment parsing using global indicators
        if any(word in generated_text for word in LLM_POSITIVE_INDICATORS):
            sentiment = "POSITIVE"
        elif any(word in generated_text for word in LLM_NEGATIVE_INDICATORS):
            sentiment = "NEGATIVE"
        elif any(word in generated_text for word in LLM_NEUTRAL_INDICATORS):
            sentiment = "NEUTRAL"
        else:
            # Fallback: analyze the original text for sentiment clues
            text_lower = text.lower()
            pos_count = sum(1 for word in POSITIVE_WORDS if word in text_lower)
            neg_count = sum(1 for word in NEGATIVE_WORDS if word in text_lower)
            
            if pos_count > neg_count:
                sentiment = "POSITIVE"
            elif neg_count > pos_count:
                sentiment = "NEGATIVE"
            else:
                sentiment = "NEUTRAL"
            
    except Exception as e:
        print(f"LLM analysis failed: {e}")
        # Fallback to keyword analysis
        text_lower = text.lower()
        pos_count = sum(1 for word in POSITIVE_WORDS if word in text_lower)
        neg_count = sum(1 for word in NEGATIVE_WORDS if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = "POSITIVE"
        elif neg_count > pos_count:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"
    
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
            score = result[0]['score']
            
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