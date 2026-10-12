"""
StockSense sentiment API.

Run from the backend folder:
    uvicorn app.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS
from app.model_service import ModelNotLoadedError, sentiment_model
from app.schemas import BatchIn, BatchOut, HealthOut, ModelInfoOut, NewsIn, SentimentOut

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    sentiment_model.load()  # loaded once at startup, not per request
    yield


app = FastAPI(
    title="StockSense Sentiment API",
    description="Classifies financial news as positive, neutral or negative.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _predict_or_503(texts: list[str]) -> list[dict]:
    try:
        return sentiment_model.predict(texts)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/health", response_model=HealthOut, tags=["system"])
def health():
    return {"status": "ok", "model_loaded": sentiment_model.loaded}


@app.get("/model/info", response_model=ModelInfoOut, tags=["system"])
def model_info():
    try:
        return {"classes": sentiment_model.classes, "metrics": sentiment_model.metrics}
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/predict", response_model=SentimentOut, tags=["prediction"])
def predict(data: NewsIn):
    if not data.news.strip():
        raise HTTPException(status_code=422, detail="news must not be blank.")
    return _predict_or_503([data.news])[0]


@app.post("/predict/batch", response_model=BatchOut, tags=["prediction"])
def predict_batch(data: BatchIn):
    if any(not t.strip() for t in data.news):
        raise HTTPException(status_code=422, detail="news items must not be blank.")
    results = _predict_or_503(data.news)
    return {"count": len(results), "results": results}

from fastapi import FastAPI, HTTPException, Query   # add Query to your existing import
from app.news_service import NewsServiceError, fetch_company_news, fetch_market_news


def _with_sentiment(articles: list[dict]) -> list[dict]:
    if not articles:
        return []
    predictions = _predict_or_503([a["headline"] for a in articles])
    return [{**a, **p} for a, p in zip(articles, predictions)]


@app.get("/news", tags=["news"])
def market_news(category: str = "general", limit: int = Query(10, ge=1, le=50)):
    try:
        articles = fetch_market_news(category, limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except NewsServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return _with_sentiment(articles)


@app.get("/news/company/{symbol}", tags=["news"])
def company_news(symbol: str, days: int = Query(7, ge=1, le=30), limit: int = Query(10, ge=1, le=50)):
    try:
        articles = fetch_company_news(symbol, days, limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except NewsServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return _with_sentiment(articles)
