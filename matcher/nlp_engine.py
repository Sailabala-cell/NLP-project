# matcher/nlp_engine.py

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk import bigrams
import string

ps = PorterStemmer()


def get_bigrams(tokens):
    return set(bigrams(tokens))


def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)

    # Remove very small words (noise like 'a', 'to', etc.)
    tokens = [word for word in tokens if len(word) > 2]

    # Stem after cleaning
    stemmed = [ps.stem(word) for word in tokens]

    return stemmed


def calculate_match_score(cv_text, jd_text):
    cv_tokens = preprocess(cv_text)
    jd_tokens = preprocess(jd_text)

    cv_set = set(cv_tokens)
    jd_set = set(jd_tokens)

    cv_bigrams = get_bigrams(cv_tokens)
    jd_bigrams = get_bigrams(jd_tokens)

    common_words = cv_set.intersection(jd_set)
    common_bigrams = cv_bigrams.intersection(jd_bigrams)

    # Coverage-based scoring (realistic)
    word_coverage = len(common_words) / max(len(jd_set), 1)
    bigram_coverage = len(common_bigrams) / max(len(jd_bigrams), 1)

    # Weighted scoring
    final_score = (word_coverage * 0.7 + bigram_coverage * 0.3) * 100

    return round(final_score, 2), list(common_words), [
        " ".join(bg) for bg in common_bigrams
    ]