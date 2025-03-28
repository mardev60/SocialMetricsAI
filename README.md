# SocialMetrics AI - API d'Analyse de Sentiments

Ce projet est une API d'analyse de sentiments pour les tweets, développée pour SocialMetrics AI. 

L'API permet d'évaluer le sentiment des tweets en fonction de leur contenu, en attribuant un score entre -1 (très négatif) et 1 (très positif).

## Fonctionnalités

- **Analyse de sentiments** : Endpoint API pour analyser le sentiment de tweets
- **Base de données** : Stockage des tweets annotés dans MySQL
- **Modèle ML avancé** : Utilisation d'un modèle d'ensemble (Random Forest, Régression Logistique, Gradient Boosting) pour prédire les sentiments avec une haute précision
- **Prétraitement du texte** : Tokenization, normalisation et gestion des contractions françaises
- **Réentraînement automatique** : Mécanisme de réentraînement hebdomadaire du modèle
- **Rapports d'évaluation** : Génération de matrices de confusion et métriques de performance

## Prérequis

- Docker et Docker Compose (méthode recommandée)
- *ou* Python 3.8+ et MySQL Server pour l'installation manuelle

## Installation et démarrage avec Docker (Recommandé)

1. Cloner le dépôt :
   ```
   git clone https://github.com/mardev60/SocialMetricsAI.git
   cd SocialMetricsAI
   ```

2. Créer les répertoires nécessaires pour le stockage des données :
   ```
   mkdir -p data/models data/reports logs reports
   ```

3. Construire et démarrer les conteneurs avec Docker Compose :
   ```
   docker-compose up -d
   ```
   Cela va:
   - Créer et démarrer un conteneur MySQL pour la base de données
   - Construire et démarrer l'application Python avec l'API Flask

4. Exécuter l'installation complète (configuration BD, génération de données et entraînement du modèle):
   ```
   docker-compose exec web python -m src.scripts.setup_all
   ```
   Cette commande configure tout automatiquement en une seule étape.

5. Redémarrer le service web pour s'assurer que le modèle est bien chargé :
   ```
   docker-compose restart web
   ```

6. L'API sera accessible à l'adresse: `http://localhost:5000`

7. Installation des ressources NLTK nécessaires :
   ```
   docker-compose exec web python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```
   Cette étape est essentielle pour permettre au modèle d'utiliser les fonctionnalités avancées de traitement du langage naturel.

## Insertion de données et entraînement du modèle avec Docker

Si vous préférez exécuter les étapes séparément ou si vous avez besoin de les répéter:

1. Pour configurer la base de données et insérer les données d'exemple:
   ```
   docker-compose exec web python -m src.database.setup_db
   ```

2. Pour générer plus de données d'entraînement et les insérer dans la base de données:
   ```
   docker-compose exec web python -m src.scripts.generate_dataset
   ```

3. Pour entraîner ou réentraîner le modèle avec les données de la base:
   ```
   docker-compose exec web python -m src.scripts.retrain
   ```

4. Vous pouvez vérifier l'état de votre base de données avec:
   ```
   docker-compose exec db mysql -usocialmetrics -psocialmetrics -e "SELECT COUNT(*) FROM socialmetrics.tweets"
   ```

## Installation manuelle (Alternative)

1. Cloner le dépôt :
   ```
   git clone https://github.com/mardev60/SocialMetricsAI.git
   cd SocialMetricsAI
   ```

2. Créer un environnement virtuel et installer les dépendances :
   ```
   python -m venv virtenv
   source virtenv/bin/activate  # Sur Windows : virtenv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configurer la base de données :
   - Modifier les paramètres de connexion dans `src/database/config.py` si nécessaire
   - Exécuter le script de configuration de la base de données :
     ```
     python -m src.database.setup_db
     ```

4. Générer plus de données d'entraînement (optionnel) :
   ```
   python -m src.scripts.generate_dataset
   ```

5. Entraîner le modèle :
   ```
   python -m src.scripts.retrain
   ```

6. Lancer l'API :
   ```
   python main.py
   ```

## Structure du projet

```
SocialMetricsAI/
├── data/
│   ├── models/           # Modèles entraînés
│   └── reports/          # Rapports d'évaluation générés
├── logs/                 # Logs de réentraînement
├── src/
│   ├── api/              # Implémentation de l'API Flask
│   │   ├── __init__.py
│   │   └── app.py
│   ├── database/         # Gestion de la base de données
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── database.py
│   │   └── setup_db.py
│   ├── models/           # Modèles de machine learning
│   │   ├── __init__.py
│   │   └── sentiment_model.py
│   ├── scripts/          # Scripts utilitaires
│   │   ├── __init__.py
│   │   ├── generate_dataset.py
│   │   ├── retrain.py
│   │   └── setup_cron.py
│   └── utils/            # Utilitaires divers
│       ├── __init__.py
│       └── evaluation_report_template.md
├── tests/                # Tests unitaires et d'intégration
│   ├── __init__.py
│   ├── test_direct.py
│   ├── test_model.py
│   └── test_sentiment.py
├── main.py               # Point d'entrée principal
├── Dockerfile            # Configuration pour construire l'image Docker
├── docker-compose.yml    # Configuration pour orchestrer les services
├── setup.py              # Configuration du package
└── requirements.txt      # Dépendances du projet
```

## Flux de travail typique

1. Configurer la base de données (`setup_db.py`)
2. Insérer des données d'exemple (`generate_dataset.py`)
3. Entraîner le modèle (`retrain.py`)
4. Lancer l'API pour analyser des tweets (`main.py`)
5. Ajouter de nouveaux tweets annotés via l'API
6. Réentraîner périodiquement le modèle

## Utilisation de l'API

### Analyser des tweets

**Endpoint** : `POST /analyze`

**Corps de la requête** :
```json
{
    "tweets": [
        "J'adore ce produit, il est fantastique !",
        "Ce service est vraiment terrible, je suis déçu."
    ]
}
```

**Réponse** :
```json
{
    "J'adore ce produit, il est fantastique !": 0.85,
    "Ce service est vraiment terrible, je suis déçu.": -0.72
}
```

### Ajouter un tweet annoté

**Endpoint** : `POST /add_tweet`

**Corps de la requête** :
```json
{
    "text": "Ce produit est incroyable !",
    "positive": 1,
    "negative": 0
}
```

### Récupérer tous les tweets

**Endpoint** : `GET /tweets`

## Rapports d'évaluation

Après chaque entraînement, le système génère :
- Des matrices de confusion pour les prédictions positives et négatives
- Un rapport détaillé avec les métriques de performance (précision, rappel, F1-score)
- Une analyse des forces, faiblesses et biais potentiels du modèle
- Des recommandations pour améliorer les performances

Ces rapports sont stockés dans le dossier `data/reports/`.

## Résolution des problèmes courants

- **Erreur de connexion à la base de données** : Vérifiez que le service MySQL est bien démarré avec `docker-compose ps` et que les variables d'environnement dans `docker-compose.yml` sont correctes. Le port MySQL interne doit être configuré à 3306 (`MYSQL_PORT=3306`).

- **Erreur "Can't connect to MySQL server"** : Si vous voyez cette erreur, vérifiez que la configuration du port MySQL est correcte dans `docker-compose.yml`. Le conteneur MySQL utilise le port 3306 en interne, même s'il est mappé à 3307 sur votre machine hôte.

- **Erreur "No such file or directory"** : Vérifiez que les répertoires nécessaires existent avec `docker-compose exec web ls -la /app/data`. Créez les répertoires manquants avec `docker-compose exec web mkdir -p /app/data/models /app/data/reports`.

- **Modèle non trouvé** : Assurez-vous d'avoir entraîné le modèle en exécutant `python -m src.scripts.retrain` dans le conteneur. Si l'erreur persiste après l'entraînement, redémarrez le service web avec `docker-compose restart web`.

- **Erreur "le modèle n'est pas chargé"** : Après avoir entraîné le modèle, redémarrez le service web avec `docker-compose restart web` pour que l'application charge le modèle nouvellement créé.

- **Erreur lors de l'insertion des données** : Vérifiez que la base de données et les tables ont été correctement créées avec `docker-compose exec db mysql -usocialmetrics -psocialmetrics -e "SHOW TABLES FROM socialmetrics"`.

- **Erreur "Resource punkt not found"** : Si vous rencontrez des erreurs liées aux ressources NLTK, exécutez les commandes de téléchargement de ressources mentionnées dans la section d'installation :
  ```
  docker-compose exec web python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
  ```

- **Performances du modèle insuffisantes** : Si les performances du modèle ne sont pas satisfaisantes pour votre cas d'usage :
  1. Ajoutez plus de données d'entraînement spécifiques à votre domaine
  2. Ajustez les poids des différents classifieurs dans le modèle d'ensemble
  3. Modifiez les paramètres de prétraitement du texte dans la fonction `preprocess_text`
  4. Consultez les rapports d'évaluation générés pour identifier les types d'erreurs les plus fréquents

## Commandes Docker utiles

- **Voir les logs de l'application** : `docker-compose logs web`
- **Voir les logs de la base de données** : `docker-compose logs db`
- **Se connecter à la base de données** : `docker-compose exec db mysql -usocialmetrics -psocialmetrics socialmetrics`
- **Exécuter une commande dans le conteneur web** : `docker-compose exec web [commande]`
- **Redémarrer uniquement l'application** : `docker-compose restart web`
- **Redémarrer tous les services** : `docker-compose restart`
- **Arrêter tous les services** : `docker-compose down`

## Licence

Ce projet est développé dans le cadre d'un TP et n'est pas sous licence spécifique.

## Performances du modèle

Le modèle actuel offre des performances de haute qualité pour l'analyse de sentiments en français :

- **Sentiment positif** :
  - Précision globale : 86.05%
  - Precision score : 90.24%
  - Recall score : 72.55%
  - F1 score : 80.43%

- **Sentiment négatif** :
  - Précision globale : 86.82%
  - Precision score : 84.71%
  - Recall score : 94.74%
  - F1 score : 89.44%

Ces performances sont le résultat des améliorations suivantes :

1. **Modèle d'ensemble** : Combinaison de plusieurs algorithmes pour une prédiction plus robuste
2. **Prétraitement avancé** : Techniques spécifiques pour le français incluant la gestion des contractions et la normalisation
3. **Équilibrage des classes** : Ajustement dynamique des poids pour gérer le déséquilibre des données
4. **Règles linguistiques** : Intégration de règles spécifiques pour les expressions françaises

## Utilisation avancée

### Analyse détaillée des sentiments

Le modèle fournit non seulement un score de sentiment, mais peut être utilisé pour une analyse plus détaillée :

```python
from src.models.sentiment_model import load_model, predict_sentiment

# Charger le modèle
models, vectorizer, embeddings = load_model()

# Analyser un texte avec détails
tweet = "Ce produit a une excellente qualité mais le prix est trop élevé"
score, details = predict_sentiment(tweet, models, vectorizer, embeddings, return_details=True)

print(f"Score global: {score}")
print(f"Probabilité positive: {details['positive_proba']}")
print(f"Probabilité négative: {details['negative_proba']}")
print(f"Caractéristiques détectées: {details['features']}")
```

### Personnalisation du modèle

Pour adapter le modèle à votre cas d'usage spécifique, vous pouvez :

1. **Ajouter des données spécifiques à votre domaine** :
   ```
   docker-compose exec web python -m src.scripts.add_custom_data --file custom_tweets.csv
   ```

2. **Ajuster les paramètres du modèle** :
   Modifier les paramètres du modèle dans `src/models/sentiment_model.py` pour ajuster le poids des différents classifieurs ou les hyperparamètres.

3. **Réentraîner après modifications** :
   ```
   docker-compose exec web python -m src.scripts.retrain
   ```