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
   - Modifier les paramètres de connexion dans `config.py`
   - Exécuter le script de configuration de la base de données :
     ```
     python setup_db.py
     ```

4. Entraîner le modèle initial :
   ```
   python model.py
   ```

5. Lancer l'API :
   ```
   python api.py
   ```

## Structure du projet

- `api.py` : Implémentation de l'API Flask
- `db.py` : Gestion de la connexion à la base de données MySQL
- `model.py` : Entraînement et évaluation du modèle de machine learning
- `retrain.py` : Script pour le réentraînement automatique du modèle
- `setup_db.py` : Configuration de la base de données
- `config.py` : Configuration de la base de données
- `reports/` : Dossier contenant les rapports d'évaluation générés
- `logs/` : Dossier contenant les logs de réentraînement

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

Ces rapports sont stockés dans le dossier `reports/`.

## Licence

Ce projet est développé dans le cadre d'un TP et n'est pas sous licence spécifique.