"""
Text cleaning shared by train.py and the API.

IMPORTANT: this must stay identical to the cleaning loop in train.py.
train.py checks this automatically after cleaning, so any drift is caught.
"""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Download stopwords the first time if they are missing
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

snow_ball = SnowballStemmer("english")

# Words kept on purpose because they matter for market sentiment
KEEP = {"not", "no", "nor", "up", "down", "over", "under", "above", "below"}
STOP_WORDS = set(stopwords.words("english")) - KEEP


def clean_text(text: str) -> str:
    """Same steps as the training loop: letters only -> lowercase -> stopwords -> stem."""
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower()
    review = review.split()
    review = [snow_ball.stem(word) for word in review if word not in STOP_WORDS]
    return " ".join(review)
