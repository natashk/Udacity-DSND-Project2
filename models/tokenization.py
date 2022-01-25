# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('words')
nltk.download('stopwords')

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def tokenize(text):
    '''
    INPUT:
    text - string, to be tokenized

    OUTPUT:
    tokens - list of normalized and lemmatized tokens
    '''

    stop_words = stopwords.words("english")

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stop words and lemmatize
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens if token not in stop_words]

    return tokens
