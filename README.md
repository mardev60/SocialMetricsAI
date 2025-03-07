# SocialMetrics AI - API d'Analyse de Sentiments

Ce projet est une API d'analyse de sentiments pour les tweets, développée pour SocialMetrics AI. 

L'API permet d'évaluer le sentiment des tweets en fonction de leur contenu, en attribuant un score entre -1 (très négatif) et 1 (très positif).

## Fonctionnalités

- **Analyse de sentiments** : Endpoint API pour analyser le sentiment de tweets
- **Base de données** : Stockage des tweets annotés dans MySQL
- **Modèle ML** : Utilisation de la régression logistique pour prédire les sentiments
- **Réentraînement automatique** : Mécanisme de réentraînement hebdomadaire du modèle
- **Rapports d'évaluation** : Génération de matrices de confusion et métriques de performance

## Prérequis

- Python 3.8+
- MySQL Server
- Bibliothèques Python (voir `requirements.txt`)

## Installation

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
   - Modifier les paramètres de connexion dans `src/database/config.py`
   - Exécuter le script de configuration de la base de données :
     ```
     python -m src.database.setup_db
     ```

4. Entraîner le modèle initial :
   ```
   python -m src.models.sentiment_model
   ```

5. Lancer l'API :
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
│   │   ├── generate_dataset_v2.py
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
├── setup.py              # Configuration du package
└── requirements.txt      # Dépendances du projet
```

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

## Licence

Ce projet est développé dans le cadre d'un TP et n'est pas sous licence spécifique.