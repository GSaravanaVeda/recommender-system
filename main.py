# main.py (moved from backend/main.py)
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os, joblib, math
from chromadb import PersistentClient
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import EmbeddingFunction
from google import genai
# load .env into environment when present (safe: .env should not be checked into VCS)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; environment variables may be set by the OS or CI
    pass

# NEW: import CORS
from fastapi.middleware.cors import CORSMiddleware
import re

# -------- Embeddings ----------
class TfidfEmbedding(EmbeddingFunction):
    def __init__(self, vec): self.vec = vec
    def __call__(self, input: list[str]):
        return self.vec.transform(input).toarray().tolist()

# -------- App bootstrap ----------
app = FastAPI(title="LLM Recommender Agent (with Evidence, Why-NOT, MMR)")

# NEW: add CORS *immediately after* creating app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # during local dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Defer heavy initialization to startup so importing this module doesn't
# fail when files/collections are missing or env vars aren't set.
vec = None
ef = None
coll = None
client = None


@app.on_event("startup")
async def startup():
    """Load model, embedding function, collection and API client at runtime.
    Any failures are caught and printed so the ASGI app can still import.
    """
    global vec, ef, coll, client
    # load vectorizer
    try:
        vec = joblib.load(os.path.join("models", "vectorizer.pkl"))
    except Exception as e:
        print("Warning: failed to load vectorizer.pkl:", e)
        vec = None

    if vec is not None:
        ef = TfidfEmbedding(vec)
        try:
            pc = PersistentClient(path="vectordb")
            try:
                coll = pc.get_collection("products", embedding_function=ef)
            except NotFoundError:
                # create the collection if it doesn't exist (matches Colab behavior)
                try:
                    coll = pc.create_collection(name="products", embedding_function=ef)
                    print("Created missing collection 'products' in vectordb")
                except Exception as e2:
                    print("Warning: failed to create vectordb collection 'products':", e2)
                    coll = None
        except Exception as e:
            print("Warning: failed to open vectordb PersistentClient:", e)
            coll = None
    else:
        ef = None

    # init genai client only if API key is present
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            client = genai.Client(api_key=api_key)
        except Exception as e:
            print("Warning: failed to initialize genai client:", e)
            client = None
    else:
        client = None


# Minimal health endpoint so uvicorn can verify the app
@app.get("/")
async def root():
    return {"status": "ok", "vectordb_loaded": bool(coll), "vectorizer_loaded": bool(vec)}


# --- Chat NLP parser + /chat endpoint ---------------------------------
class ChatReq(BaseModel):
    message: str
    diversity_lambda: float = 0.6
    return_why_not: bool = True


_price_pat = re.compile(r"(?:under|below|less than|<=?)\s*(₹?\s*\d+[.,]?\d*)", re.I)
_price_num = re.compile(r"\d+[.,]?\d*")
_rating_pat = re.compile(r"(?:rating|stars?)\s*(?:above|over|>=?)\s*(\d+(?:\.\d+)?)", re.I)
_include_pat = re.compile(r"(?:include|must have|with)\s+([a-z0-9 ,\-]+)", re.I)
_exclude_pat = re.compile(r"(?:exclude|without|no)\s+([a-z0-9 ,\-]+)", re.I)


def _parse_message(msg: str):
    text = msg.strip()

    # max price
    max_price = None
    m = _price_pat.search(text)
    if m:
        n = _price_num.search(m.group(1).replace("₹", "").replace(",", ""))
        if n:
            try:
                max_price = float(n.group())
            except:
                pass

    # min rating
    min_rating = None
    r = _rating_pat.search(text)
    if r:
        try:
            min_rating = float(r.group(1))
        except:
            pass

    # include terms
    include = None
    mi = _include_pat.search(text)
    if mi:
        include = [w.strip() for w in mi.group(1).split(",") if w.strip()]

    # exclude terms
    exclude = None
    me = _exclude_pat.search(text)
    if me:
        exclude = [w.strip() for w in me.group(1).split(",") if w.strip()]

    # interests = full message (good signal for vector search)
    interests = text

    return interests, max_price, min_rating, include, exclude


@app.post("/chat")
def chat_recommend(req: ChatReq):
    interests, max_price, min_rating, include, exclude = _parse_message(req.message)

    # reuse your recommend flow programmatically
    tmp = RecommendReq(
        interests=interests,
        max_price=max_price,
        min_rating=min_rating,
        include=include,
        exclude=exclude,
        diversity_lambda=req.diversity_lambda,
        return_why_not=req.return_why_not,
    )
    return recommend(tmp)


# --- Recommendation endpoint (simple safe stub) -----------------
class RecommendReq(BaseModel):
    interests: str
    max_price: Optional[float] = None
    min_rating: Optional[float] = None
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None
    diversity_lambda: float = 0.5
    return_why_not: bool = False


@app.post("/recommend")
@app.post("/recommend/")
def recommend(req: RecommendReq):
    """Return a minimal, predictable recommendations payload so the
    Streamlit frontend can work even if the real retrieval pipeline
    (vectordb / model / genai) isn't configured.
    """
    # If vectordb and vectorizer are available, you can replace this
    # stub with the real retrieval/rerank logic. For now return a
    # small example response.
    results = [
        {
            "title": "Eco Bamboo Utensil Set",
            "price": 29.99,
            "rating": 4.5,
            "score": 95.2,
            "reason": "Matches your interest in 'bamboo' and eco-friendly materials",
        },
        {
            "title": "Bamboo Cutting Board",
            "price": 24.5,
            "rating": 4.6,
            "score": 88.3,
            "reason": "High relevance to 'kitchen' and 'bamboo'",
        },
    ]

    why_not = []
    if req.return_why_not:
        why_not = [
            {"title": "Cheap Plastic Utensils", "why_not": "Contains 'plastic' (excluded)"},
        ]

    # indicate which mode produced this response so it's obvious in the UI
    used = "gemini" if client is not None else "stub"
    return {"results": results, "why_not": why_not, "mode": used}
