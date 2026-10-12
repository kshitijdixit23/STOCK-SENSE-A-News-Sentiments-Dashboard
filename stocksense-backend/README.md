# StockSense Backend (FastAPI)

Financial news sentiment API (positive / neutral / negative).
TF-IDF + Logistic Regression pipeline, tuned with GridSearchCV.

## Structure

```
stocksense-backend/
├── app/
│   ├── main.py            # FastAPI app and endpoints
│   ├── model_service.py   # loads model, runs predictions
│   ├── preprocess.py      # clean_text() shared with training
│   ├── schemas.py         # request/response models
│   └── config.py          # paths, CORS, limits
├── data/                  # put all-data.csv here
├── models/                # trained model is saved here
├── train.py               # your training code + saves the model
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## 1. Train (once, or whenever you change the data/cleaning)

Copy `all-data.csv` into `data/`, then:

```bash
python train.py
```

This also checks that `clean_text()` matches your cleaning loop, then saves
`models/sentiment_model.joblib` and `models/metrics.json`.

## 2. Run the API

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs to test every endpoint.

## Endpoints

| Method | Path             | Purpose                              |
|--------|------------------|--------------------------------------|
| GET    | `/health`        | Is the server up / model loaded      |
| GET    | `/model/info`    | Classes and training metrics         |
| POST   | `/predict`       | Sentiment for one headline           |
| POST   | `/predict/batch` | Sentiment for up to 100 headlines    |

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"news": "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn"}'
```

```json
{
  "sentiment": "positive",
  "confidence": 0.87,
  "probabilities": {"negative": 0.04, "neutral": 0.09, "positive": 0.87}
}
```

## Notes

- If you change the cleaning in `train.py`, change `app/preprocess.py` too, then retrain.
  `train.py` stops with an error if the two ever differ.
- Use the same scikit-learn version for training and serving.
- React dev servers on ports 5173 and 3000 are allowed by CORS. Change with
  the `CORS_ORIGINS` environment variable (comma separated).
