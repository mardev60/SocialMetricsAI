from flask import Flask, request, jsonify
import random
from db import insert_tweet, get_all_tweets

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()

    if not data or "tweets" not in data:
        return jsonify({"error": "liste de tweets requise"}), 400

    tweets = data["tweets"]

    if not isinstance(tweets, list) or not all(isinstance(t, str) for t in tweets):
        return jsonify({"error": "format invalide, tweets doit être une liste de chaînes"}), 400

    results = {}
    for tweet in tweets:
        score = round(random.uniform(-1, 1), 2)
        positive = 1 if score > 0 else 0
        negative = 1 if score < 0 else 0

        insert_tweet(tweet, positive, negative)

        results[tweet] = score

    return jsonify(results)

@app.route('/tweets', methods=['GET'])
def get_tweets():
    tweets = get_all_tweets()
    return jsonify(tweets)

if __name__ == '__main__':
    app.run(debug=True)