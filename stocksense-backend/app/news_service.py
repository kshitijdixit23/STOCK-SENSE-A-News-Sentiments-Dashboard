from datetime import date, datetime, timedelta, timezone
import re

import httpx

from app.config import FINNHUB_API_KEY

BASE_URL = "https://finnhub.io/api/v1"
MARKET_CATEGORIES = {"general", "forex", "crypto", "merger"}


class NewsServiceError(RuntimeError):
    """Raised when headlines can't be fetched from Finnhub."""


def _get(path: str, params: dict) -> list[dict]:
    if not FINNHUB_API_KEY:
        raise NewsServiceError("FINNHUB_API_KEY is missing. Add it to your .env file.")
    try:
        r = httpx.get(
            f"{BASE_URL}{path}",
            params=params,
            headers={"X-Finnhub-Token": FINNHUB_API_KEY},  # key stays out of the URL
            timeout=10,
        )
        r.raise_for_status()
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code in (401, 403):
            msg = "Finnhub rejected the API key."
        elif code == 429:
            msg = "Finnhub rate limit reached. Try again in a minute."
        else:
            msg = f"Finnhub returned an error ({code})."
        raise NewsServiceError(msg) from exc
    except httpx.RequestError as exc:
        raise NewsServiceError("Could not reach Finnhub.") from exc
    return r.json()


def _format(articles: list[dict], limit: int) -> list[dict]:
    out = []
    for a in articles:
        if not a.get("headline"):
            continue
        ts = a.get("datetime")
        out.append(
            {
                "headline": a["headline"],
                "summary": a.get("summary", ""),
                "source": a.get("source", ""),
                "url": a.get("url", ""),
                "image": a.get("image", ""),
                "published_at": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None,
            }
        )
        if len(out) >= limit:
            break
    return out


def fetch_market_news(category: str = "general", limit: int = 10) -> list[dict]:
    if category not in MARKET_CATEGORIES:
        raise ValueError(f"category must be one of {sorted(MARKET_CATEGORIES)}")
    return _format(_get("/news", {"category": category}), limit)


def fetch_company_news(symbol: str, days: int = 7, limit: int = 10) -> list[dict]:
    symbol = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z.\-]{1,10}", symbol):
        raise ValueError("symbol looks invalid (example: AAPL).")
    today = date.today()
    params = {
        "symbol": symbol,
        "from": (today - timedelta(days=days)).isoformat(),
        "to": today.isoformat(),
    }
    return _format(_get("/company-news", params), limit)