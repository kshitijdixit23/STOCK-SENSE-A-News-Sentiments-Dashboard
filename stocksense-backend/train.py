"""
Train the sentiment model and save it for the API.

Run from the backend folder:
    python train.py

Your original code is kept as it was (cleaning loop, split, pipeline, grid search,
evaluation). Additions: saving the model + metrics, and a check that the shared
clean_text() gives exactly the same output as the loop.
"""
# importing the libraries
import json
import re

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from app.config import DATA_PATH, METRICS_PATH, MODEL_PATH
from app.preprocess import clean_text

nltk.download("stopwords", quiet=True)

# import the pandas csv
messages = pd.read_csv(
    DATA_PATH, encoding="Latin-1", names=["sentiment", "news"], header=None
)

# using the stemming*
snow_ball = SnowballStemmer("english")
keep = {"not", "no", "nor", "up", "down", "over", "under", "above", "below"}
stop_words = set(stopwords.words("english")) - keep
corpus = []

# Cleaning the data
for i in range(0, len(messages)):
    review = re.sub("[^a-zA-Z]", " ", messages["news"][i])
    review = review.lower()
    review = review.split()
    review = [snow_ball.stem(word) for word in review if word not in stop_words]
    review = " ".join(review)
    corpus.append(review)

# NEW: safety check -> the API's clean_text() must match this loop exactly
mismatches = [
    i for i, text in enumerate(messages["news"]) if clean_text(text) != corpus[i]
]
if mismatches:
    raise SystemExit(
        f"clean_text() differs from the training loop on {len(mismatches)} rows "
        f"(first row: {mismatches[0]}). Fix app/preprocess.py before training."
    )
print("Cleaning check passed: training loop and clean_text() match.")


# train & test
from sklearn.model_selection import train_test_split

y = messages["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(
    corpus, y, test_size=0.2, random_state=42, stratify=y
)

# using pipeline for more accuracy*
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline(
    [
        ("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]
)


# grid search
from sklearn.model_selection import GridSearchCV

params = {
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "clf__C": [0.5, 1, 5, 10],
}
grid = GridSearchCV(pipe, params, scoring="f1_macro", cv=5)
grid.fit(X_train, y_train)

print("Best Settings:", grid.best_params_)
print("Best CV Macro-F1", grid.best_score_)


# predict & evaluate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = grid.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# NEW: save the best pipeline + metrics so the API can load them
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(grid.best_estimator_, MODEL_PATH)

report = classification_report(y_test, y_pred, output_dict=True)
metrics = {
    "best_params": {
        k: list(v) if isinstance(v, tuple) else v for k, v in grid.best_params_.items()
    },
    "cv_macro_f1": float(grid.best_score_),
    "test_accuracy": float(accuracy_score(y_test, y_pred)),
    "test_macro_f1": float(report["macro avg"]["f1-score"]),
    "classes": [str(c) for c in grid.best_estimator_.classes_],
    "train_size": len(X_train),
    "test_size": len(X_test),
}
METRICS_PATH.write_text(json.dumps(metrics, indent=2))
print(f"Model saved to {MODEL_PATH}")
print(f"Metrics saved to {METRICS_PATH}")


'''
# Word Embedding usage
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(max_features=10000,ngram_range=(1,2),min_df=2,sublinear_tf=True)
X = tfidf.fit_transform(corpus).toarray() #text → numbers array

#train the classifier
from sklearn.naive_bayes import MultinomialNB
model= MultinomialNB()
model.fit(X_train, y_train) #numbers → learn

'''
