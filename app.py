from flask import Flask, request, jsonify
import os
from sentimentAnalysisController.predict import predict_sentiment
from neuralNetworkController.predict import predict_NN

app = Flask(__name__)

@app.route('/model/predict', methods=['POST'])
def sentiment_analysis():
    data = request.get_json(force=True)
    text = data['text']
    sentiment_score = predict_sentiment(text)
    fer_label = data['fer_label']
    print(sentiment_score[0,0], fer_label)

    if sentiment_score is None or fer_label is None:
        return jsonify({'error': 'Invalid input'}), 400
    prediction = predict_NN(sentiment_score[0], fer_label)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    port = int(os.getenv("PORT", 8081))  # Use PORT from environment or default to 8081
    app.run(host='0.0.0.0', port=port)
