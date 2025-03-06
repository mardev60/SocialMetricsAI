from flask import Flask, request, jsonify
from db import insert_tweet, get_all_tweets
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    with open("sentiment_model.pkl", "rb") as model_file:
        models = pickle.load(model_file)

    with open("vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    
    print("modèle et vectoriseur chargés avec succès")
except FileNotFoundError:
    print("modèle ou vectoriseur non trouvé. Veuillez exécuter model.py pour entraîner le modèle")
    models = None
    vectorizer = None

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    if models is None or vectorizer is None:
        return jsonify({"error": "le modèle n'est pas chargé"}), 500

    data = request.get_json()

    if not data or "tweets" not in data:
        return jsonify({"error": "liste de tweets requise"}), 400

    tweets = data["tweets"]

    if not isinstance(tweets, list) or not all(isinstance(t, str) for t in tweets):
        return jsonify({"error": "format invalide, tweets doit être une liste de chaînes"}), 400

    X_input = vectorizer.transform(tweets)
    
    positive_scores = models['positive'].predict_proba(X_input)[:, 1]
    negative_scores = models['negative'].predict_proba(X_input)[:, 1]

    final_scores = positive_scores - negative_scores

    results = {tweet: round(float(score), 2) for tweet, score in zip(tweets, final_scores)}

    return jsonify(results)

@app.route('/tweets', methods=['GET'])
def get_tweets():
    tweets = get_all_tweets()
    return jsonify(tweets)

if __name__ == '__main__':
    app.run(debug=True)