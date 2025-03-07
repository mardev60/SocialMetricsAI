import pickle
from model import analyze_sentiment_text

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

print("\nRésultats de l'analyse de sentiments :")
print("-" * 70)
print(f"{'Tweet':<40} | {'Score':<10} | {'Sentiment':<10}")
print("-" * 70)

for tweet in test_tweets:
    score = analyze_sentiment_text(tweet, models, vectorizer)
    
    sentiment = "POSITIF" if score > 0 else "NÉGATIF" if score < 0 else "NEUTRE"
    
    display_tweet = tweet[:37] + "..." if len(tweet) > 40 else tweet
    
    print(f"{display_tweet:<40} | {score:>10.2f} | {sentiment:<10}")

print("-" * 70) 