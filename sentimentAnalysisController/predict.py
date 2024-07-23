# import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .preprocess import clean_text

model = load_model('model/analysis_sentiment_model.h5')
tokenizer = Tokenizer()

def predict_sentiment(text):
    preprocessed_text = clean_text(text)
    tokenizer.fit_on_texts([preprocessed_text])
    sequences = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequences = pad_sequences(sequences, maxlen=16)
    prediction = model.predict(padded_sequences)
    return prediction
