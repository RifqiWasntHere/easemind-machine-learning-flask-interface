from sklearn.preprocessing import LabelEncoder

# FER emotion scores mapping
fer_emotion_scores = {
    'angry': 0.2,
    'disgust': 0.3,
    'fear': 0.4,
    'happy': 1.0,
    'neutral': 0.5,
    'sad': 0.1,
    'surprise': 0.8
}

# Initialize the LabelEncoder and fit it with the final labels
final_labels = ['stressed', 'slightly stressed', 'neutral', 'okay', 'happy']
label_encoder = LabelEncoder()
label_encoder.fit(final_labels)

# Function to get FER score
def get_fer_score(fer_label):
    return fer_emotion_scores.get(fer_label, 0)

# Weighted average function
def weighted_average(sentiment_score, fer_score, sentiment_weight=0.6, fer_weight=0.4):
    combined_score = (sentiment_score * sentiment_weight) + (fer_score * fer_weight)
    return combined_score
