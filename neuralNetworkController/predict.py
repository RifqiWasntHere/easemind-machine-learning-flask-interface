import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from .preprocess import get_fer_score, weighted_average, label_encoder

# Load the trained model
model = load_model('model/combined_model.h5')

def predict_NN(sentiment_score, fer_label):
    fer_score = get_fer_score(fer_label)

    # Calculate combined score
    combined_score = weighted_average(sentiment_score, fer_score)

    # Predict using the model
    combined_score = np.array([[combined_score]])
    prediction = model.predict(combined_score)
    predicted_class = tf.argmax(prediction, axis=1).numpy()[0]

    # Decode the predicted class to the final label
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_label
