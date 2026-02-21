
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from nltk import bigrams
from nltk import pos_tag



ps = PorterStemmer()

def get_bigrams(tokens):
    return list(bigrams(tokens))

def filter_nouns(tokens):
    tagged = pos_tag(tokens)
    nouns = [word for word, tag in tagged if tag.startswith('NN')]
    return nouns

def preprocess(text):
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Stem
    stemmed = [ps.stem(word) for word in tokens]
    
    return stemmed

def calculate_match_score(cv_text, jd_text):
    cv_words = filter_nouns(preprocess(cv_text))
    jd_words = filter_nouns(preprocess(jd_text))
    
    cv_bigrams = get_bigrams(cv_words)
    jd_bigrams = get_bigrams(jd_words)
    
    common_words = set(cv_words).intersection(set(jd_words))
    common_bigrams = set(cv_bigrams).intersection(set(jd_bigrams))
    
    word_score = len(common_words)
    bigram_score = len(common_bigrams) * 2  # give higher weight
    
    total_score = word_score + bigram_score
    
    final_score = total_score / len(set(jd_words)) * 100
    
    return final_score, common_words, common_bigrams

# ------------------------
# TEST DATA
# ------------------------

cv = """
Frontend developer skilled in React, JavaScript, Django and REST APIs.
Built scalable backend services.
"""

jd = """
Looking for a developer with Django and React experience.
Must understand REST API development.
"""

score, matched_words, matched_bigrams = calculate_match_score(cv, jd)

print("Match Score:", round(score, 2), "%")
print("Matched Words:", matched_words)
print("Matched Bigrams:", matched_bigrams)
print("Matched Bigrams:", [' '.join(b) for b in matched_bigrams])