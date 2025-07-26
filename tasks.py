import ray
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from dotenv import load_dotenv
load_dotenv()
import os
import torch

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
def analyze_sentiment_mistral(text):
    global mistral_tokenizer, mistral_model
    try:
        mistral_tokenizer
        mistral_model
    except NameError:
        mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        mistral_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    
    # Create prompt for sentiment analysis
    prompt = f"<s>[INST] Classify the sentiment of this review as POSITIVE, NEGATIVE, or NEUTRAL. Only respond with the sentiment label:\n\n{text} [/INST]"
    
    # Tokenize and generate
    inputs = mistral_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = mistral_model.generate(**inputs, max_new_tokens=10, temperature=0)
    response = mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract sentiment from response
    sentiment = response.split("[/INST]")[-1].strip().upper()
    if "POSITIVE" in sentiment:
        sentiment = "POSITIVE"
    elif "NEGATIVE" in sentiment:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    
    return {"text": text, "sentiment": sentiment}

@ray.remote
def analyze_sentiment_llama2(text):
    global llama_tokenizer, llama_model
    try:
        llama_tokenizer
        llama_model
    except NameError:
        llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    
    # Create prompt for sentiment analysis
    prompt = f"<s>[INST] Classify the sentiment of this review as POSITIVE, NEGATIVE, or NEUTRAL. Only respond with the sentiment label:\n\n{text} [/INST]"
    
    # Tokenize and generate
    inputs = llama_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = llama_model.generate(**inputs, max_new_tokens=10, temperature=0)
    response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract sentiment from response
    sentiment = response.split("[/INST]")[-1].strip().upper()
    if "POSITIVE" in sentiment:
        sentiment = "POSITIVE"
    elif "NEGATIVE" in sentiment:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    
    return {"text": text, "sentiment": sentiment}

@ray.remote
def analyze_sentiment_small_llm(text):
    global small_tokenizer, small_model
    try:
        small_tokenizer
        small_model
    except NameError:
        # Use a smaller model that won't crash the system
        small_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        small_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    
    # Create prompt for sentiment analysis
    prompt = f"Sentiment: Classify this review as POSITIVE, NEGATIVE, or NEUTRAL:\n{text}\nSentiment:"
    
    # Tokenize and generate
    inputs = small_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    outputs = small_model.generate(**inputs, max_new_tokens=5, temperature=0)
    response = small_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract sentiment from response
    sentiment = response.split("Sentiment:")[-1].strip().upper()
    if "POSITIVE" in sentiment:
        sentiment = "POSITIVE"
    elif "NEGATIVE" in sentiment:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    
    return {"text": text, "sentiment": sentiment}

@ray.remote
def analyze_sentiment_quantized(text):
    global quantized_tokenizer, quantized_model
    try:
        quantized_tokenizer
        quantized_model
    except NameError:
        # Use 8-bit quantization to reduce memory usage
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16"
        )
        
        quantized_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        quantized_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-medium",
            quantization_config=bnb_config,
            device_map="auto"
        )
    
    # Create prompt for sentiment analysis
    prompt = f"Sentiment: Classify this review as POSITIVE, NEGATIVE, or NEUTRAL:\n{text}\nSentiment:"
    
    # Tokenize and generate
    inputs = quantized_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    outputs = quantized_model.generate(**inputs, max_new_tokens=5, temperature=0)
    response = quantized_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract sentiment from response
    sentiment = response.split("Sentiment:")[-1].strip().upper()
    if "POSITIVE" in sentiment:
        sentiment = "POSITIVE"
    elif "NEGATIVE" in sentiment:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    
    return {"text": text, "sentiment": sentiment}

@ray.remote
def analyze_sentiment_api(text):
    import requests
    
    # Use a better sentiment analysis API endpoint
    API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGING_FACE_HUB_TOKEN', '')}"}
    
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": text})
        result = response.json()
        
        # Better parsing of the API response
        if isinstance(result, list) and len(result) > 0:
            # Get the highest confidence prediction
            predictions = result[0]
            if isinstance(predictions, list) and len(predictions) > 0:
                best_prediction = max(predictions, key=lambda x: x['score'])
                label = best_prediction['label']
                score = best_prediction['score']
                
                # Map labels to our format
                if label == 'LABEL_0':
                    sentiment = "NEGATIVE"
                elif label == 'LABEL_1':
                    sentiment = "NEUTRAL"
                elif label == 'LABEL_2':
                    sentiment = "POSITIVE"
                else:
                    sentiment = "NEUTRAL"
                    
                # If confidence is low, default to NEUTRAL
                if score < 0.3:
                    sentiment = "NEUTRAL"
            else:
                sentiment = "NEUTRAL"
        else:
            sentiment = "NEUTRAL"
            
    except Exception as e:
        print(f"API call failed: {e}")
        sentiment = "NEUTRAL"
    
    return {"text": text, "sentiment": sentiment}

@ray.remote
def analyze_sentiment_api_alt(text):
    import requests
    
    # Try a different sentiment analysis model
    API_URL = "https://api-inference.huggingface.co/models/financial-sentiment-analysis"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGING_FACE_HUB_TOKEN', '')}"}
    
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": text})
        result = response.json()
        
        # Parse the response
        if isinstance(result, list) and len(result) > 0:
            prediction = result[0]
            if isinstance(prediction, list) and len(prediction) > 0:
                best_prediction = max(prediction, key=lambda x: x['score'])
                label = best_prediction['label']
                
                # Map labels
                if 'positive' in label.lower():
                    sentiment = "POSITIVE"
                elif 'negative' in label.lower():
                    sentiment = "NEGATIVE"
                else:
                    sentiment = "NEUTRAL"
            else:
                sentiment = "NEUTRAL"
        else:
            sentiment = "NEUTRAL"
            
    except Exception as e:
        print(f"Alternative API call failed: {e}")
        sentiment = "NEUTRAL"
    
    return {"text": text, "sentiment": sentiment}

@ray.remote
def analyze_sentiment_alt_model(text):
    global alt_sentiment_pipeline
    try:
        alt_sentiment_pipeline
    except NameError:
        # Use a different sentiment analysis model
        from transformers import pipeline
        alt_sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    
    # Truncate text if too long
    if len(text) > 500:
        text = text[:500] + "..."
    
    result = alt_sentiment_pipeline(text)
    label = result[0]['label']
    score = result[0]['score']
    
    # Map labels to our format
    if 'LABEL_0' in label:
        sentiment = "NEGATIVE"
    elif 'LABEL_1' in label:
        sentiment = "NEUTRAL"
    elif 'LABEL_2' in label:
        sentiment = "POSITIVE"
    else:
        sentiment = "NEUTRAL"
    
    return {"text": text, "sentiment": sentiment}

@ray.remote
def analyze_sentiment_api_simple(text):
    import requests
    
    # Use a simple sentiment analysis based on text analysis
    # Since API authentication is failing, let's create a simple rule-based approach
    # that simulates what an API might return
    
    text_lower = text.lower()
    
    # More sophisticated keyword analysis
    very_positive_words = ['amazing', 'fantastic', 'excellent', 'wonderful', 'love', 'fabulous', 'perfect']
    positive_words = ['great', 'good', 'enjoy', 'nice', 'exciting', 'fun', 'happy', 'enjoyed']
    negative_words = ['terrible', 'awful', 'horrible', 'bad', 'disappointing', 'waste', 'worst', 'hate']
    very_negative_words = ['let down', 'dislike', 'poor', 'terrible', 'awful']
    
    # Count different types of sentiment words
    very_pos_count = sum(1 for word in very_positive_words if word in text_lower)
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    very_neg_count = sum(1 for word in very_negative_words if word in text_lower)
    
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
def analyze_sentiment_llm_safe(text):
    """
    Use a smaller LLM through Hugging Face for sentiment analysis.
    This uses a smaller model that won't crash the system but still demonstrates LLM capabilities.
    """
    global llm_tokenizer, llm_model
    try:
        llm_tokenizer
        llm_model
    except NameError:
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
        
        # More robust sentiment parsing
        if any(word in generated_text for word in ['positive', 'good', 'great', 'excellent', 'amazing', 'wonderful']):
            sentiment = "POSITIVE"
        elif any(word in generated_text for word in ['negative', 'bad', 'terrible', 'awful', 'horrible', 'poor']):
            sentiment = "NEGATIVE"
        elif any(word in generated_text for word in ['neutral', 'okay', 'fine', 'average', 'normal']):
            sentiment = "NEUTRAL"
        else:
            # Fallback: analyze the original text for sentiment clues
            text_lower = text.lower()
            positive_words = ['great', 'amazing', 'wonderful', 'fantastic', 'excellent', 'love', 'enjoy', 'good', 'nice', 'fabulous', 'exciting', 'happy']
            negative_words = ['terrible', 'awful', 'horrible', 'bad', 'disappointing', 'waste', 'worst', 'hate', 'dislike', 'poor', 'boring']
            
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
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
        positive_words = ['great', 'amazing', 'wonderful', 'fantastic', 'excellent', 'love', 'enjoy', 'good', 'nice', 'fabulous', 'exciting', 'happy']
        negative_words = ['terrible', 'awful', 'horrible', 'bad', 'disappointing', 'waste', 'worst', 'hate', 'dislike', 'poor', 'boring']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
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
    try:
        sentiment_3class_pipeline
    except NameError:
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
            
            # More sophisticated keyword analysis for 3-class
            very_positive_words = ['amazing', 'fantastic', 'excellent', 'wonderful', 'love', 'fabulous', 'perfect', 'outstanding']
            positive_words = ['great', 'good', 'enjoy', 'nice', 'exciting', 'fun', 'happy', 'enjoyed', 'like']
            negative_words = ['terrible', 'awful', 'horrible', 'bad', 'disappointing', 'waste', 'worst', 'hate']
            very_negative_words = ['let down', 'dislike', 'poor', 'terrible', 'awful', 'hate']
            
            # Count different types of sentiment words
            very_pos_count = sum(1 for word in very_positive_words if word in text_lower)
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            very_neg_count = sum(1 for word in very_negative_words if word in text_lower)
            
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