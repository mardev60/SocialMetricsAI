import pickle
from model import preprocess_french_text, detect_sentiment_keywords, analyze_sentiment_text

test_tweets = [
    "C'est super sympa",
    "Horrible je n'ai pas aimé",
    "Le film était vraiment mal réalisé, j'ai perdu mon temps",
    "J'ai adoré ce restaurant, c'était excellent !",
    "Très satisfait de mon achat",
    "Application facile à utiliser et pratique",
    "Produit de bonne qualité, je recommande",
    "Service client catastrophique, à éviter absolument",
    "Expérience décevante, je ne reviendrai pas",
    "Film ennuyeux et trop long",
    "Je n'ai pas aimé ce film",
    "Ce n'est pas terrible",
    "Ce n'est pas mauvais",
    "Je n'ai pas détesté",
    "Ce restaurant n'est pas du tout recommandable"
]

try:
    with open("sentiment_model.pkl", "rb") as model_file:
        models = pickle.load(model_file)

    with open("vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    
    print("Modèle et vectoriseur chargés avec succès")
except FileNotFoundError:
    print("Modèle ou vectoriseur non trouvé")
    exit(1)

preprocessed_tweets = [preprocess_french_text(tweet) for tweet in test_tweets]

X_input = vectorizer.transform(preprocessed_tweets)

positive_scores = models['positive'].predict_proba(X_input)[:, 1]
negative_scores = models['negative'].predict_proba(X_input)[:, 1]

keyword_scores = [detect_sentiment_keywords(tweet) for tweet in test_tweets]

print("\nRésultats de l'analyse de sentiments :")
print("-" * 70)
print(f"{'Tweet':<40} | {'Score Modèle':<12} | {'Score Mots-clés':<15} | {'Score Final':<10} | {'Sentiment':<10}")
print("-" * 70)

for i, tweet in enumerate(test_tweets):
    model_score = (positive_scores[i] - negative_scores[i]) * 0.2
    
    if keyword_scores[i] != 0:
        adjusted_score = model_score + (0.5 * keyword_scores[i])
    else:
        adjusted_score = model_score * 2
    
    sentiment = "POSITIF" if adjusted_score > 0 else "NÉGATIF" if adjusted_score < 0 else "NEUTRE"
    
    display_tweet = tweet[:37] + "..." if len(tweet) > 40 else tweet
    
    print(f"{display_tweet:<40} | {model_score:>10.2f}  | {keyword_scores[i]:>13}  | {adjusted_score:>10.2f} | {sentiment:<10}")

print("-" * 70)
print("Analyse des problèmes potentiels :")
print("-" * 70)

for i, tweet in enumerate(test_tweets):
    model_score = positive_scores[i] - negative_scores[i]
    keyword_score = keyword_scores[i]
    
    if (model_score > 0 and keyword_score < 0) or (model_score < 0 and keyword_score > 0):
        print(f"Incohérence pour: \"{tweet}\"")
        print(f"  - Score modèle: {model_score:.2f} ({'positif' if model_score > 0 else 'négatif'})")
        print(f"  - Score mots-clés: {keyword_score} ({'positif' if keyword_score > 0 else 'négatif'})")
        print(f"  - Cause possible: Le modèle et la détection par mots-clés sont en désaccord")
        print("-" * 70) 