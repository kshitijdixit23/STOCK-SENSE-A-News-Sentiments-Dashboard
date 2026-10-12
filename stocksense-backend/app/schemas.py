"""Request / response shapes. FastAPI uses these for validation and the /docs page."""
from pydantic import BaseModel, Field

from app.config import MAX_BATCH_SIZE, MAX_TEXT_LENGTH


class NewsIn(BaseModel):
    news: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        examples=["Operating profit rose to EUR 13.1 mn from EUR 8.7 mn a year earlier."],
    )


class BatchIn(BaseModel):
    news: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        examples=[["Profits rose sharply.", "Shares fell after weak earnings."]],
    )


class SentimentOut(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict[str, float]


class BatchOut(BaseModel):
    count: int
    results: list[SentimentOut]


class HealthOut(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoOut(BaseModel):
    classes: list[str]
    metrics: dict | None = None
