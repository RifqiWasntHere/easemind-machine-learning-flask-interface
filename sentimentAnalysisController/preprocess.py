import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd

stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

indo_slang_word = pd.read_csv("https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv")
normalization_dict = dict(zip(indo_slang_word['slang'], indo_slang_word['formal']))

def normalize_text(text):
    words = word_tokenize(text)
    normalized_words = [normalization_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)

def clean_text(text):
    text = text.lower()
    text = normalize_text(text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)
