import os
import re
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import uvicorn
import wikipedia
from textblob import TextBlob

app = FastAPI(title="Advanced NLP Chatbot")

# Create templates directory if it doesn't exist
os.makedirs("templates", exist_ok=True)
templates = Jinja2Templates(directory="templates")

class ChatbotModel:
    def __init__(self):
        self.intents = [
            {"patterns": ["hi", "hello", "hey", "good morning"], "response": "Hello! How can I help you today?"},
            {"patterns": ["what is your name", "who are you"], "response": "I am an Advanced NLP Chatbot. I can answer casual questions, analyze your sentiment, and search Wikipedia for general knowledge!"},
            {"patterns": ["how do you work", "how were you built"], "response": "I use TF-IDF and Cosine Similarity for basic matching, TextBlob for sentiment analysis, and the Wikipedia API for answering factual questions outside of my dataset."},
            {"patterns": ["what is nlp", "tell me about nlp", "what does nlp stand for"], "response": "NLP stands for Natural Language Processing. It is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language."},
            {"patterns": ["bye", "goodbye", "see you later", "exit"], "response": "Goodbye! Have a great day!"},
            {"patterns": ["thank you", "thanks", "thanks a lot"], "response": "You're very welcome!"},
            {"patterns": ["you are stupid", "idiot", "hate you"], "response": "I apologize if I did anything wrong. I am still learning!"},
            {"patterns": ["joke", "tell me a joke", "make me laugh"], "response": "Why do programmers prefer dark mode? Because light attracts bugs!"}
        ]
        self.vectorizer = TfidfVectorizer(lowercase=True)
        self.corpus = []
        self.responses = []
        
        for intent in self.intents:
            for pattern in intent["patterns"]:
                self.corpus.append(pattern)
                self.responses.append(intent["response"])
        
        if self.corpus:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def extract_search_query(self, text: str) -> str:
        # Remove common question phrases to get the core topic for Wikipedia
        text = text.lower()
        phrases_to_remove = ["what is", "who is", "tell me about", "do you know", "search for", "who was", "what are"]
        for phrase in phrases_to_remove:
            if text.startswith(phrase):
                return text.replace(phrase, "", 1).strip()
        return text.strip()

    def get_reply(self, text: str) -> str:
        if not text.strip():
            return "Please say something."
            
        # 1. Sentiment Analysis
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        
        prefix = ""
        if sentiment < -0.4:
            prefix = "You seem a bit upset. "
        elif sentiment > 0.6:
            prefix = "It sounds like you're in a great mood! "

        # 2. Intent matching via TF-IDF
        vec = self.vectorizer.transform([text])
        sims = cosine_similarity(vec, self.tfidf_matrix)
        max_idx = np.argmax(sims)
        max_score = sims[0][max_idx]
        
        if max_score >= 0.4:
             return prefix + self.responses[max_idx]
        
        # 3. Fallback: Wikipedia Search
        query = self.extract_search_query(text)
        if query and len(query) > 2:
            try:
                # Limit to 2 sentences for a concise answer
                summary = wikipedia.summary(query, sentences=2, auto_suggest=False)
                return prefix + f"I found this on Wikipedia:\n{summary}"
            except wikipedia.exceptions.DisambiguationError as e:
                # If there are multiple meanings, just show the first few options
                options = ", ".join(e.options[:3])
                return prefix + f"That term could mean multiple things. Did you mean: {options}?"
            except wikipedia.exceptions.PageError:
                pass
            except Exception:
                pass

        return prefix + "I'm not exactly sure what you mean, and I couldn't find it on Wikipedia. Could you rephrase?"

bot = ChatbotModel()

class MessageRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    reply = bot.get_reply(req.message)
    return {"reply": reply}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
