import requests
import json

API_URL = "http://localhost:5000/analyze"

test_tweets = [
    "C'est super sympa",
    "Horrible je n'ai pas aimé",
    "Le film était vraiment mal réalisé, j'ai perdu mon temps",
    "J'ai adoré ce restaurant, c'était excellent !",
    "Service client catastrophique, à éviter absolument",
    "Produit de bonne qualité, je recommande",
    "Expérience décevante, je ne reviendrai pas",
    "Très satisfait de mon achat",
    "Film ennuyeux et trop long",
    "Application facile à utiliser et pratique"
]

data = {"tweets": test_tweets}

try:
    response = requests.post(API_URL, json=data)
    
    if response.status_code == 200:
        results = response.json()
        
        print("Résultats de l'analyse de sentiments :")
        print("-" * 50)
        
        for tweet, score in results.items():
            sentiment = "POSITIF" if score > 0 else "NÉGATIF" if score < 0 else "NEUTRE"
            print(f"Tweet: \"{tweet}\"")
            print(f"Score: {score} ({sentiment})")
            print("-" * 50)
    else:
        print(f"Erreur: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"Erreur lors de la connexion à l'API: {e}") 