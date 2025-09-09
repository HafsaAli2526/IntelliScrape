# nlp_tasks.py    
from transformers import pipeline

# Initialize transformers pipelines once
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def summarize_text(text, max_length=130, min_length=30):
    # Truncate the input text to ~1024 tokens worth (a rough guess: 4000 chars)
    safe_text = text.strip()[:4000]

    try:
        summary = summarizer(
            safe_text, max_length=max_length, min_length=min_length, do_sample=False
        )
        return summary[0]['summary_text']
    
    except Exception as e:
        return f"❌ Error in summarization: {str(e)}"
    

def analyze_sentiment(text):
   # Truncate to ~2048 characters (~512 tokens max)
    safe_text = text.strip()[:2048]
    try:
        return sentiment_analyzer(safe_text)[0]
    except Exception as e:
        return {"label": "error", "score": 0.0, "error": str(e)}

def run_qa(client, context, question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
    ]
    try:
        response = client.chat_completion(messages=messages, max_tokens=512)

        # Check that response and choices exist
        if hasattr(response, "choices") and response.choices and len(response.choices) > 0:
            return response.choices[0].message["content"]
        else:
            return "⚠️ No answer returned by the model. Try simplifying the input or retrying."

    except Exception as e:
        return f"❌ Error in Q&A: {str(e)}"
    

    