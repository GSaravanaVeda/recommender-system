# backend/main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os, json, ast, joblib, re, traceback
from pathlib import Path
from dotenv import load_dotenv
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction
from google import genai

# --- env (.env next to this file) ---
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# --- embeddings wrapper ---
class TfidfEmbedding(EmbeddingFunction):
    def __init__(self, vec): self.vec = vec
    def __call__(self, input: list[str]):
        return self.vec.transform(input).toarray().tolist()

app = FastAPI(title="LLM Recommender Agent (Chat + Evidence + MMR)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---- global exception handler: return JSON instead of 500 ----
@app.exception_handler(Exception)
async def all_ex_handler(request: Request, exc: Exception):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return JSONResponse({"error": str(exc), "traceback": tb}, status_code=200)

# lazy globals
vec = ef = coll = None
client = None

@app.on_event("startup")
async def startup():
    global vec, ef, coll, client
    # vectorizer
    try:
        vec = joblib.load("models/vectorizer.pkl"); ef = TfidfEmbedding(vec)
    except Exception as e:
        print("WARN vectorizer:", e); vec = None; ef = None
    # vectordb
    try:
        if ef:
            coll = PersistentClient(path="vectordb").get_collection(
                "products", embedding_function=ef
            )
    except Exception as e:
        print("WARN vectordb:", e); coll = None
    # gemini (GEMINI_API_KEY or GOOGLE_API_KEY)
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        try: client = genai.Client(api_key=key)
        except Exception as e:
            print("WARN genai init:", e); client = None

@app.get("/")
def root():
    return {"status":"ok","vectordb_loaded":bool(coll),"vectorizer_loaded":bool(vec),
            "mode":"gemini" if client else "fallback"}

@app.get("/health")
def health():
    return {"ok":True,"has_key":bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
            "vectordb_loaded":bool(coll),"vectorizer_loaded":bool(vec),
            "mode":"gemini" if client else "fallback"}

# ---------- helpers ----------
def _filter(items: List[Dict[str, Any]], mx, mn, inc, exc):
    kept, dropped = [], []
    for it in items:
        reasons=[]
        p = it.get("price")
        if mx is not None and (p is None or p > mx): reasons.append(f"over budget (price={p} > {mx})")
        r = float(it.get("rating") or 0.0)
        if mn is not None and r < mn: reasons.append(f"low rating ({r} < {mn})")
        text = (" ".join([str(it.get(k,"")) for k in ("title","category","features","material")])).lower()
        if inc:
            miss=[w for w in inc if w.lower() not in text]
            if miss: reasons.append(f"missing terms {miss}")
        if exc:
            pres=[w for w in exc if w.lower() in text]
            if pres: reasons.append(f"contains excluded {pres}")
        (dropped if reasons else kept).append((it,reasons) if reasons else it)
    return kept, dropped

def _cos(a,b):
    import numpy as np
    a,b = np.array(a), np.array(b); na,nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na==0 or nb==0 else float(a@b/(na*nb))

def _mmr(cands, scores, embs, top_n=5, lamb=0.6):
    sel, seli = [], []
    if not scores: scores=[50.0]*len(cands)
    smin,smax=min(scores),max(scores)
    ns=[0.5]*len(scores) if smax==smin else [(s-smin)/(smax-smin) for s in scores]
    while len(sel)<min(top_n,len(cands)):
        best_i,best_v=None,-1e9
        for i in range(len(cands)):
            if i in seli: continue
            rel=ns[i]
            div=0.0 if not seli else max(_cos(embs[i], embs[j]) for j in seli)
            v=lamb*rel-(1-lamb)*div
            if v>best_v: best_v,best_i=v,i
        seli.append(best_i); sel.append(cands[best_i])
    return sel, seli

def _pack(items, limit=30):
    return [{
        "id": str(it.get("id")),
        "title": it.get("title",""),
        "category": it.get("category",""),
        "price": it.get("price",""),
        "rating": it.get("rating",""),
        "features": (it.get("features","") or "")[:180],
        "material": it.get("material","")
    } for it in items[:limit]]

def _prompt(user_interest, compact, k):
    return f"""
You are a recommender. User wants: "{user_interest}".
Score each candidate (0-100) for relevance, price fit, and quality (rating).
Return ONLY JSON array of top {k}:
[{{"id":"...","score":0-100,"reason":"ONE short sentence citing ONLY fields: title/category/price/rating/features/material."}}]
Do not invent facts.
CANDIDATES:
{compact}
""".strip()

def _why_not_filters(drop):
    return [{"id":str(it.get("id")),"title":it.get("title",""),
             "price":it.get("price"),"rating":it.get("rating"),
             "why_not":"; ".join(reasons)} for it,reasons in drop]

def _why_not_scores(allmap, kept_ids):
    sel=set(kept_ids); out=[]
    for _id,it in allmap.items():
        if _id not in sel:
            out.append({"id":_id,"title":it.get("title",""),
                        "price":it.get("price"),"rating":it.get("rating"),
                        "why_not":"lower relevance score than selected items"})
    return out

def _safe_json(text: str):
    t=(text or "").strip()
    if not t: return []
    if t.startswith("```"):
        t=t.strip("`")
        if "\n" in t: t=t.split("\n",1)[1]
    try: return json.loads(t)
    except Exception:
        try: return ast.literal_eval(t)
        except Exception: return []

# ---------- schemas ----------
class RecommendReq(BaseModel):
    interests: str
    max_price: Optional[float] = None
    min_rating: Optional[float] = None
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None
    diversity_lambda: float = 0.6
    return_why_not: bool = True

# ---------- core recommend ----------
@app.post("/recommend")
@app.post("/recommend/")
def recommend(req: RecommendReq):
    # no DB guard
    if not coll:
        return {"results": [], "why_not": [], "mode": "fallback-no-vdb"}

    # 1) retrieve with explicit includes (prevents KeyErrors)
    res = coll.query(
        query_texts=[req.interests],
        n_results=50,
        include=["ids","documents","metadatas","embeddings"]
    )
    items = res.get("metadatas", [[]])[0]
    ids   = [str(x) for x in res.get("ids", [[]])[0]]
    docs  = res.get("documents", [[]])[0]
    embs  = res.get("embeddings", [None])[0]
    if embs is None and docs:
        embs = ef(docs)
    if not items:
        return {"results": [], "why_not": [], "mode": "no-candidates"}

    # 2) filters
    kept, dropped = _filter(items, req.max_price, req.min_rating, req.include, req.exclude)
    if not kept:
        return {"results": [], "why_not": _why_not_filters(dropped), "mode": "filtered-out"}

    id2item = {str(it.get("id")): it for it in kept}
    id2emb  = {i:e for i,e in zip(ids, embs or []) if i in id2item}

    # 3) LLM rerank (safe)
    compact = _pack(kept, 30)
    used, data = "fallback", []
    if client and compact:
        try:
            resp = client.models.generate_content(model="gemini-2.5-flash",
                                                  contents=_prompt(req.interests, compact, 10))
            data = _safe_json(getattr(resp, "text", "") or "")
            if isinstance(data, dict): data = data.get("items", [])
            used = "gemini" if data else "fallback"
        except Exception as e:
            print("WARN gemini call:", e); used = "fallback"

    if not data:  # fallback
        kept_sorted = sorted(kept, key=lambda x: (-(x.get("rating") or 0), x.get("price") or 1e9))[:10]
        data = [{"id": str(x.get("id")), "score": 60,
                 "reason": "High rating and reasonable price based on metadata."} for x in kept_sorted]

    scored=[]
    for row in data:
        rid = str(row.get("id"))
        base = id2item.get(rid)
        if not base: continue
        scored.append({
            "id": rid, "title": base.get("title",""),
            "price": base.get("price"), "rating": base.get("rating"),
            "category": base.get("category",""), "features": base.get("features",""),
            "material": base.get("material",""),
            "score": float(row.get("score",0)), "reason": row.get("reason","")
        })
    if not scored:
        return {"results": [], "why_not": _why_not_filters(dropped), "mode": used}

    scores=[s["score"] for s in scored]
    embs_s=[id2emb.get(s["id"]) for s in scored]
    zipped=[(s,sc,e) for s,sc,e in zip(scored, scores, embs_s) if e is not None]
    if not zipped:
        final = scored[:5]; sel_ids=[s["id"] for s in final]
    else:
        s2,sc2,e2 = zip(*zipped)
        final,_ = _mmr(list(s2), list(sc2), list(e2), top_n=5, lamb=req.diversity_lambda)
        sel_ids=[s["id"] for s in final]

    why_not=[]
    if req.return_why_not:
        why_not += _why_not_filters(dropped)
        all_kept_map = {s["id"]: id2item[s["id"]] for s in scored}
        why_not += _why_not_scores(all_kept_map, sel_ids)

    return {"results": final, "why_not": why_not[:20], "mode": used}

# ---------- chat to structured ----------
class ChatReq(BaseModel):
    message: str
    diversity_lambda: float = 0.6
    return_why_not: bool = True

_price_pat   = re.compile(r"(?:under|below|less than|<=?)\s*(₹?\s*\d+[.,]?\d*)", re.I)
_price_num   = re.compile(r"\d+[.,]?\d*")
_rating_pat  = re.compile(r"(?:rating|stars?)\s*(?:above|over|>=?)\s*(\d+(?:\.\d+)?)", re.I)
_include_pat = re.compile(r"(?:include|must have|with)\s+([a-z0-9 ,\-]+)", re.I)
_exclude_pat = re.compile(r"(?:exclude|without|no)\s+([a-z0-9 ,\-]+)", re.I)

def _parse_msg(msg: str):
    t = msg.strip()
    max_price = None
    if (m := _price_pat.search(t)):
        n = _price_num.search(m.group(1).replace("₹","").replace(",",""))
        if n:
            try: max_price=float(n.group())
            except: pass
    min_rating=None
    if (r := _rating_pat.search(t)):
        try: min_rating=float(r.group(1))
        except: pass
    include=None
    if (mi := _include_pat.search(t)):
        include=[w.strip() for w in mi.group(1).split(",") if w.strip()]
    exclude=None
    if (me := _exclude_pat.search(t)):
        exclude=[w.strip() for w in me.group(1).split(",") if w.strip()]
    interests=t
    return interests, max_price, min_rating, include, exclude

@app.post("/chat")
def chat(req: ChatReq):
    interests, max_price, min_rating, include, exclude = _parse_msg(req.message)
    body = RecommendReq(
        interests=interests, max_price=max_price, min_rating=min_rating,
        include=include, exclude=exclude, diversity_lambda=req.diversity_lambda,
        return_why_not=req.return_why_not
    )
    return recommend(body)

@app.get("/debug/parse")
def dbg_parse(q: str):
    i, mp, mr, inc, exc = _parse_msg(q)
    return {"interests":i,"max_price":mp,"min_rating":mr,"include":inc,"exclude":exc}
