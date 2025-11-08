# backend/main.py - Simple LLM Recommender Chatbot with Real Product URLs
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
import requests
from urllib.parse import quote
import json

# Load environment variables
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

app = FastAPI(title="LLM Recommender")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini client
client = None

@app.on_event("startup")
async def startup():
    global client
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            print("✓ Gemini API ready")
        except Exception as e:
            print(f"✗ Gemini error: {e}")
    else:
        print("✗ No API key")

@app.get("/")
def root():
    return {"status": "ok", "service": "LLM Recommender", "ai_ready": bool(client)}

@app.get("/health")
def health():
    return {"ok": True, "gemini": bool(client)}

# Simple chat request
class ChatRequest(BaseModel):
    message: str

def search_products(query: str, max_results: int = 5):
    """Generate real, working product URLs"""
    products = []
    
    try:
        # Clean and encode the search query
        search_query = quote(query.strip())
        
        # Amazon India - these URLs work and show real product listings
        amazon_url = f"https://www.amazon.in/s?k={search_query}&ref=nb_sb_noss"
        
        # Flipkart - these URLs work and show real product listings  
        flipkart_url = f"https://www.flipkart.com/search?q={search_query}&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off"
        
        # Myntra for fashion/lifestyle items
        myntra_url = f"https://www.myntra.com/{search_query.replace('+', '-')}"
        
        # Add multiple working links
        products = [
            {
                "name": f"Amazon India - {query}",
                "url": amazon_url,
                "platform": "Amazon"
            },
            {
                "name": f"Flipkart - {query}",
                "url": flipkart_url,
                "platform": "Flipkart"
            },
            {
                "name": f"Myntra - {query}",
                "url": myntra_url,
                "platform": "Myntra"
            }
        ]
        
    except Exception as e:
        print(f"Product search error: {e}")
    
    return products

@app.post("/chat")
def chat(req: ChatRequest):
    """Chat with real product URLs"""
    if not client:
        return {
            "response": "Error: AI service not available. Please check API key.",
            "error": True
        }
    
    try:
        # First, get AI to extract product keywords
        extract_prompt = f"""Extract the main product keywords from this user request: "{req.message}"

Return ONLY the product name/type in 2-4 words. Examples:
User: "I want jute bags" -> "jute bags"
User: "recommend kitchen items under 1000" -> "kitchen items"
User: "wireless earphones" -> "wireless earphones"

Just the keywords, nothing else."""

        keyword_response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=extract_prompt
        )
        
        keywords = keyword_response.text.strip() if hasattr(keyword_response, 'text') else req.message
        print(f"Extracted keywords: {keywords}")
        
        # Get real product URLs
        product_links = search_products(keywords)
        
        # Create response with real, clickable links
        main_prompt = f"""You are a helpful shopping assistant. The user asked: "{req.message}"

Here are REAL, WORKING shopping links I've generated:
Amazon: {product_links[0]['url']}
Flipkart: {product_links[1]['url']}
Myntra: {product_links[2]['url']}

Create 3-5 specific product recommendations. For EACH product:
1. Give it a realistic, specific product name
2. Add estimated price in ₹
3. Use ONE of the real URLs above (rotate between them)
4. Write a brief description

Format EXACTLY like this (use the ACTUAL URLs provided):

### 1. [Specific Product Name]({product_links[0]['url']})
**Price:** ₹XXX - ₹XXX  
Brief description and why it's good.

### 2. [Another Product Name]({product_links[1]['url']})
**Price:** ₹XXX - ₹XXX  
Brief description.

Continue for 3-5 products. Make product names realistic and specific. USE THE EXACT URLS I PROVIDED ABOVE."""

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=main_prompt
        )
        
        text = response.text if hasattr(response, 'text') else str(response)
        
        return {
            "response": text,
            "error": False
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error: {error_msg}")
        
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return {
                "response": "⏳ The AI service is currently busy. Please wait a moment and try again.",
                "error": True
            }
        
        return {
            "response": f"Sorry, I encountered an error. Please try again.",
            "error": True
        }
