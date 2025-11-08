# LLM Recommender System

A smart recommendation system that uses Google's Gemini API and ChromaDB for product recommendations. The system processes natural language queries and returns personalized product suggestions based on various criteria like price, rating, and features.

## Features

- Natural language processing for product queries
- Price and rating-based filtering
- Product feature matching
- Diversity-aware recommendations using MMR
- Real-time chat interface

## Setup

1. Clone the repository:
```bash
git clone https://github.com/GSaravanaVeda/recommender-system.git
cd recommender-system
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
.\.venv\Scripts\Activate  # Windows
source .venv/bin/activate  # Linux/Mac

pip install -U fastapi[standard] google-genai chromadb joblib python-dotenv numpy streamlit requests
```

3. Configure the environment:
- Copy `backend/.env.example` to `backend/.env`
- Add your Gemini API key to `backend/.env`:
```
GEMINI_API_KEY=your_api_key_here
```

4. Start the services:

Backend (from project root):
```bash
cd backend
uvicorn main:app --reload --port 8000
```

Frontend (in another terminal, from project root):
```bash
streamlit run frontend/app.py
```

5. Open in your browser:
- Chat UI: http://localhost:8501
- API docs: http://localhost:8000/docs

## API Endpoints

- `GET /health` - Service health check
- `POST /chat` - Chat interface for natural language queries
- `POST /recommend` - Direct recommendation endpoint
- `GET /debug/parse` - Debug endpoint to test query parsing

## Architecture

- FastAPI backend with Gemini integration
- Streamlit frontend for chat interface
- ChromaDB for vector similarity search
- TF-IDF for product embeddings
- MMR for recommendation diversity

## Environment Variables

- `GEMINI_API_KEY` (required) - Google Gemini API key
- Also accepts `GOOGLE_API_KEY` as fallback