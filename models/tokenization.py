# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def tokenize(text):
    # tokenize text
    #tokens = ne_chunk(pos_tag(word_tokenize(text)))
    stop_words = stopwords.words("english")

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stop words and lemmatize
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens if token not in stop_words]

    return tokens
