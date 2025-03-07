from flask import Flask, request, jsonify
from src.database.database import insert_tweet, get_all_tweets
import pickle
import os
from src.models.sentiment_model import analyze_sentiment_text

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "models", "vectorizer.pkl")

try:
    with open(MODEL_PATH, "rb") as model_file:
        models = pickle.load(model_file)

    with open(VECTORIZER_PATH, "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    
    print("modèle et vectoriseur chargés avec succès")
except FileNotFoundError:
    print("modèle ou vectoriseur non trouvé. Veuillez exécuter src/models/sentiment_model.py pour entraîner le modèle")
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

    if not isinstance(tweets, list) and isinstance(tweets, dict):
        tweets = list(tweets.keys())
    elif not isinstance(tweets, list) or not all(isinstance(t, str) for t in tweets):
        return jsonify({"error": "format invalide, tweets doit être une liste de chaînes"}), 400

    results = {}
    for tweet in tweets:
        score = analyze_sentiment_text(tweet, models, vectorizer)
        results[tweet] = score

    return jsonify(results)

@app.route('/tweets', methods=['GET'])
def get_tweets():
    tweets = get_all_tweets()
    return jsonify(tweets)

if __name__ == '__main__':
    app.run(debug=True)