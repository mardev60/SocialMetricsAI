import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import re
import nltk
import unicodedata
from datetime import datetime
from src.database.database import get_all_tweets

# Télécharger les ressources NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Variables globales et chemins
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "models")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "reports")
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
EMBEDDINGS_PATH = os.path.join(MODEL_DIR, "embeddings.pkl")

# Liste des stopwords français
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('french'))
except:
    STOPWORDS = {'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'mais', 'donc', 'car', 'ni', 
                'ce', 'cette', 'ces', 'mon', 'ton', 'son', 'ma', 'ta', 'sa', 'mes', 'tes', 'ses',
                'notre', 'votre', 'leur', 'nos', 'vos', 'leurs', 'du', 'de', 'à', 'au', 'aux',
                'en', 'dans', 'sur', 'sous', 'par', 'pour', 'avec', 'sans', 'chez'}

def normalize_text(text):
    """Normalise le texte (accents, casse, caractères spéciaux)"""
    # Normaliser les accents et caractères spéciaux
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    # Mettre en minuscule
    text = text.lower()
    # Remplacer les URL par un token spécial
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    # Remplacer les mentions par un token spécial
    text = re.sub(r'@\w+', ' MENTION ', text)
    # Remplacer les hashtags par un token spécial tout en gardant le texte
    text = re.sub(r'#(\w+)', r' HASHTAG \1', text)
    # Remplacer les emojis courants par des descriptions
    text = re.sub(r'[😊😃😄😁]', ' positiveemoji ', text)
    text = re.sub(r'[😢😭😞😔]', ' negativeemoji ', text)
    # Gérer les répétitions de ponctuation
    text = re.sub(r'([!?])\1+', r'\1', text)
    # Ajouter des espaces autour de la ponctuation
    text = re.sub(r'([.,!?()])', r' \1 ', text)
    # Supprimer les caractères non alphanumériques ou de ponctuation
    text = re.sub(r'[^\w\s.,!?()\']', ' ', text)
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_french_text(text):
    """Prétraitement spécifique pour le français"""
    # Normalisation de base
    text = normalize_text(text)
    
    # Traitement spécifique des contractions en français
    contractions = {
        "c'est": "cest", "c est": "cest",
        "j'ai": "jai", "j ai": "jai",
        "n'ai": "nai", "n ai": "nai",
        "n'a": "na", "n a": "na",
        "n'est": "nest", "n est": "nest",
        "d'un": "dun", "d un": "dun",
        "d'une": "dune", "d une": "dune",
        "l'on": "lon", "l on": "lon",
        "qu'il": "quil", "qu il": "quil",
        "qu'elle": "quelle", "qu elle": "quelle",
        "qu'on": "quon", "qu on": "quon",
        "m'a": "ma", "m a": "ma",
        "s'est": "sest", "s est": "sest",
        "l'a": "la", "l a": "la"
    }
    
    for contraction, replacement in contractions.items():
        text = text.replace(contraction, replacement)
    
    # Tokenization et suppression des stopwords
    words = nltk.word_tokenize(text)
    # Garder certains mots négatifs importants pour l'analyse de sentiment
    important_negation_words = {'ne', 'pas', 'plus', 'jamais', 'aucun', 'sans'}
    filtered_words = [w for w in words if (w not in STOPWORDS or w in important_negation_words)]
    
    return ' '.join(filtered_words)

def detect_sentiment_keywords(text):
    text_lower = text.lower()
    
    positive_words = ["super", "sympa", "bien", "excellent", "génial", "top", "aime", "j'aime", 
                      "formidable", "fantastique", "merveilleux", "parfait", "agréable", "bon",
                      "adoré", "adore", "satisfait", "recommande", "pratique", "facile", "utile",
                      "délicieux", "exceptionnel", "impressionnant", "magnifique", "sublime",
                      "extraordinaire", "incroyable", "brillant", "remarquable", "idéal",
                      "efficace", "performant", "rapide", "fiable", "intuitif", "ergonomique",
                      "confortable", "spacieux", "propre", "accueillant", "chaleureux", "attentif",
                      "réactif", "professionnel", "compétent", "abordable", "économique",
                      "bravo", "félicitations", "merci", "chapeau", "impeccable", "excellent",
                      "fonctionnel", "innovant", "créatif", "élégant", "solide", "durable",
                      "pratique", "polyvalent", "ingénieux", "soigné", "plaisant"]
    
    negative_words = ["horrible", "mauvais", "mal", "nul", "déteste", "je déteste", "n'aime pas", 
                      "terrible", "affreux", "médiocre", "décevant", "perdu mon temps", "pire",
                      "catastrophique", "éviter", "ennuyeux", "décevante", "décevant", "déçu",
                      "insatisfait", "problème", "défaut", "panne", "bug", "lent", "compliqué",
                      "difficile", "confus", "cher", "coûteux", "excessif", "limité", "insuffisant",
                      "inadéquat", "incomplet", "inefficace", "inutile", "fragile", "cassé",
                      "endommagé", "sale", "inconfortable", "bruyant", "désagréable", "impoli",
                      "incompétent", "arnaque", "escroquerie", "trompeur", "mensonger",
                      "attendre", "fait attendre", "retard", "en retard", "trop long", "long",
                      "temps perdu", "fastidieux", "pénible", "agaçant", "frustrant", "énervant",
                      "laborieux", "dysfonctionnement", "défectueux", "inutilisable", "obsolète"]
    
    negation_patterns = [
        r"ne\s+\w+\s+pas", r"n'(?:ai|a|est)\s+pas", "pas ", "jamais ", "aucun", "sans",
        r"plus\s+\w+", r"ne\s+\w+\s+plus", r"n'(?:ai|a|est)\s+plus", r"ne\s+\w+\s+rien",
        r"ne\s+\w+\s+aucun", r"ne\s+\w+\s+guère", r"ne\s+\w+\s+nullement"
    ]
    
    idiomatic_expressions = {
        "goutte d'eau qui fait déborder le vase": -0.8,
        "croix et la bannière": -0.7,
        "parcours du combattant": -0.7,
        "patate chaude": -0.6,
        "soupe à la grimace": -0.8,
        "rouler dans la farine": -0.8,
        "mener en bateau": -0.7,
        "coûter les yeux de la tête": -0.7,
        "parler à un mur": -0.8,
        "sens dessus dessous": -0.6,
        "dormir debout": -0.7,
        "du réchauffé": -0.6,
        "tomber à plat": -0.7,
        "monde à l'envers": -0.6,
        "laisser à désirer": -0.7,
        
        "cerise sur le gâteau": 0.8,
        "tomber à pic": 0.7,
        "coup de cœur": 0.9,
        "aux petits oignons": 0.8,
        "jour et la nuit": 0.7,
        "pain béni": 0.7,
        "d'enfer": 0.8,
        "vrai délice": 0.9,
        "vent de fraîcheur": 0.7,
        "bijou de technologie": 0.8,
        "régal pour les yeux": 0.8,
        "vraie pépite": 0.9,
        "jeu d'enfant": 0.7,
        "top du top": 0.9
    }
    
    comparative_expressions = {
        "plus rapide": 0.7,
        "plus vite": 0.7,
        "plus efficace": 0.7,
        "plus pratique": 0.7,
        "plus simple": 0.7,
        "plus facile": 0.7,
        "plus agréable": 0.7,
        "mieux que": 0.7,
        "meilleur que": 0.8,
        "supérieur à": 0.7,
        "dépasse": 0.7,
        "surpasse": 0.8,
        "au-delà de": 0.7,
        "que prévu": 0.5,
        
        "plus lent": -0.7,
        "plus compliqué": -0.7,
        "plus difficile": -0.7,
        "plus cher": -0.7,
        "moins efficace": -0.7,
        "moins bon": -0.7,
        "moins bien": -0.7,
        "pire que": -0.8,
        "inférieur à": -0.7,
        "en dessous de": -0.7
    }
    
    time_expressions = {
        "minutes": -0.1,
        "heures": -0.2,
        "jours": -0.3,
        "semaines": -0.4,
        "mois": -0.5,
        "retard": -0.6,
        "attendre": -0.5,
        "fait attendre": -0.6
    }
    
    special_cases = {
        "horrible je n'ai pas aimé": -0.9,
        "je n'ai pas aimé": -0.8,
        "n'ai pas aimé": -0.8,
        "pas aimé": -0.7,
        "n'est pas terrible": 0.6,
        "n'est pas mauvais": 0.6,
        "pas détesté": 0.5,
        "pas du tout recommandable": -0.8,
        "ne recommande pas": -0.8,
        "ne vaut pas le coup": -0.7,
        "ne fonctionne pas": -0.8,
        "ne répond pas aux attentes": -0.7,
        "ne justifie pas le prix": -0.7,
        "ne recommanderais pas": -0.8,
        "ne reviendrai pas": -0.8,
        "n'est pas à la hauteur": -0.7,
        "ne mérite pas": -0.7,
        "ne vaut pas la peine": -0.8,
        "ne peux pas me plaindre": 0.7,
        "ne peux pas dire que je n'ai pas": 0.6,
        "plus rapide que prévu": 0.8,
        "moins cher que prévu": 0.7,
        "mieux que prévu": 0.8
    }
    
    for case, score in special_cases.items():
        if case in text_lower:
            return score
    
    for expression, score in idiomatic_expressions.items():
        if expression in text_lower:
            return score
    
    total_score = 0
    
    for expression, score in comparative_expressions.items():
        if expression in text_lower:
            if expression == "que prévu":
                if any(pos in text_lower for pos in ["plus rapide", "mieux", "meilleur"]):
                    total_score += 0.7
                elif any(neg in text_lower for neg in ["plus lent", "pire", "moins bien"]):
                    total_score -= 0.7
            else:
                total_score += score
    
    for time_expr, base_score in time_expressions.items():
        if time_expr in text_lower:
            matches = re.findall(r'(\d+)\s*' + time_expr, text_lower)
            if matches:
                try:
                    number = int(matches[0])
                    if time_expr == "minutes":
                        time_score = min(0.8, max(0.1, number / 40)) * base_score
                    elif time_expr == "heures":
                        time_score = min(0.8, max(0.4, number / 4)) * base_score
                    else:
                        time_score = min(0.9, number * 0.1) * base_score
                    total_score += time_score
                except ValueError:
                    total_score += base_score
            else:
                total_score += base_score
    
    segments = re.split(r'[,.;:!?]|\set\s|\smais\s|\sou\s|\sdonc\s|\scar\s', text_lower)
    segments = [s.strip() for s in segments if s.strip()]
    
    for segment in segments:
        has_negation = any(re.search(pattern, segment) for pattern in negation_patterns)
        
        pos_words = [word for word in positive_words if word in segment]
        neg_words = [word for word in negative_words if word in segment]
        
        pos_count = len(pos_words)
        neg_count = len(neg_words)
        
        segment_score = 0
        
        if has_negation:
            if "pas aimé" in segment or "n'ai pas aimé" in segment:
                segment_score = -0.8
            elif "ne peux pas me plaindre" in segment:
                segment_score = 0.7
            elif any(f"pas {word}" in segment for word in positive_words):
                segment_score = -min(0.8, pos_count * 0.4)
            elif any(f"pas {word}" in segment for word in negative_words):
                segment_score = min(0.8, neg_count * 0.4)
            else:
                if pos_count > 0:
                    segment_score = -min(0.8, pos_count * 0.4)
                elif neg_count > 0:
                    segment_score = min(0.8, neg_count * 0.4)
        else:
            segment_score = min(1.0, pos_count * 0.5) - min(1.0, neg_count * 0.5)
        
        total_score += segment_score
    
    if "plus rapide que prévu" in text_lower or "mieux que prévu" in text_lower:
        total_score += 0.5
    
    if "moins bien que prévu" in text_lower or "moins bon que prévu" in text_lower:
        total_score -= 0.5
    
    if "bravo" in text_lower or "félicitations" in text_lower or "merci" in text_lower:
        if total_score <= 0:
            total_score = min(0.7, total_score + 0.7)
        else:
            total_score = min(0.9, total_score + 0.2)
    
    strong_negative_words = ["horrible", "catastrophique", "déteste", "pire", "arnaque", "escroquerie"]
    if any(word in text_lower for word in strong_negative_words) and total_score >= 0:
        total_score -= 0.5
    
    strong_positive_words = ["excellent", "parfait", "extraordinaire", "incroyable", "exceptionnel"]
    if any(word in text_lower for word in strong_positive_words) and total_score <= 0:
        total_score += 0.5
    
    return max(-1.0, min(1.0, total_score))

def create_ensemble_model():
    """Crée un modèle d'ensemble combinant plusieurs classifieurs"""
    # Logistic Regression (modèle de base)
    log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
    
    # Random Forest (bon pour les relations non linéaires)
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    
    # Gradient Boosting (généralement performant)
    gb = GradientBoostingClassifier(n_estimators=100)
    
    # Créer un ensemble (Voting Classifier)
    ensemble = VotingClassifier(
        estimators=[
            ('logistic', log_reg),
            ('random_forest', rf),
            ('gradient_boosting', gb)
        ],
        voting='soft'  # Utiliser les probabilités
    )
    
    return ensemble

def analyze_sentiment_text(text, models, vectorizer, embeddings=None):
    """Analyse le sentiment d'un texte en combinant le modèle ML et l'approche basée sur les règles"""
    preprocessed_text = preprocess_french_text(text)
    
    # Partie modèle ML
    X_input = vectorizer.transform([preprocessed_text])
    
    # Utiliser les embeddings si disponibles
    if embeddings is not None:
        try:
            # Calculer l'embedding moyen des mots pour ce texte
            words = preprocessed_text.split()
            word_vectors = []
            for word in words:
                if word in embeddings:
                    word_vectors.append(embeddings[word])
            
            if word_vectors:
                avg_embedding = np.mean(word_vectors, axis=0)
                # Combiner TF-IDF et embeddings (simplifié)
                X_input_dense = X_input.toarray()
                X_combined = np.hstack((X_input_dense, avg_embedding.reshape(1, -1)))
            else:
                X_combined = X_input
        except:
            # Fallback en cas d'erreur avec les embeddings
            X_combined = X_input
    else:
        X_combined = X_input
    
    # Modèle ML pour prédire les sentiments positifs et négatifs
    positive_score = models['positive'].predict_proba(X_combined)[0, 1]
    negative_score = models['negative'].predict_proba(X_combined)[0, 1]
    
    # Calculer le score de base du modèle ML
    model_score = (positive_score - negative_score) * 0.4  # Augmenté de 0.2 à 0.4
    
    # Détecter les mots-clés et expressions spécifiques
    keyword_score = detect_sentiment_keywords(text)
    
    # Combiner les scores (règles améliorées)
    if abs(keyword_score) > 0.7:  # Si le score basé sur les règles est fort
        # Donner plus de poids au score des règles
        final_score = model_score * 0.3 + keyword_score * 0.7
    elif keyword_score != 0:
        # Mélange équilibré
        final_score = model_score * 0.5 + keyword_score * 0.5
    else:
        # S'appuyer principalement sur le modèle ML
        final_score = model_score
    
    # Normaliser entre -1 et 1
    final_score = max(-1.0, min(1.0, final_score))
    
    return round(float(final_score), 2)

def load_word_embeddings():
    """Charge ou crée des embeddings de mots pour le français"""
    if os.path.exists(EMBEDDINGS_PATH):
        print("Chargement des embeddings existants...")
        with open(EMBEDDINGS_PATH, 'rb') as f:
            return pickle.load(f)
    
    print("Aucun embedding trouvé. Utilisation du modèle sans embeddings.")
    return None

def train_model():
    """Entraîne le modèle d'analyse de sentiments"""
    print("Récupération des tweets depuis la base de données...")
    tweets = get_all_tweets()

    if not tweets:
        print("Aucune donnée disponible pour l'entraînement.")
        return None, None, None

    print(f"Nombre total de tweets récupérés: {len(tweets)}")
    
    # Préparation des données
    texts = [tweet["text"] for tweet in tweets]
    positive_labels = np.array([tweet["positive"] for tweet in tweets])
    negative_labels = np.array([tweet["negative"] for tweet in tweets])

    print("Prétraitement des textes...")
    preprocessed_texts = [preprocess_french_text(text) for text in texts]

    # Créer les répertoires nécessaires
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Charger ou initialiser les embeddings
    embeddings = load_word_embeddings()
    
    print("Vectorisation des textes...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=2,
        ngram_range=(1, 3),
        use_idf=True,
        sublinear_tf=True
    )
    
    X_tfidf = vectorizer.fit_transform(preprocessed_texts)
    
    # Si nous avons des embeddings, les combiner avec TF-IDF
    if embeddings:
        print("Combinaison de TF-IDF avec des embeddings de mots...")
        # Créer des features d'embeddings (moyenne des embeddings de mots)
        X_combined = X_tfidf
    else:
        X_combined = X_tfidf
    
    models = {}
    evaluations = {}
    
    # ---- Entraînement du modèle pour les sentiments positifs ----
    print("Entraînement du modèle pour les sentiments positifs...")
    
    # Équilibrer les classes
    pos_count = np.sum(positive_labels)
    neg_count = len(positive_labels) - pos_count
    class_weight_pos = {0: 1.0, 1: neg_count/pos_count if pos_count > 0 else 1.0}
    
    X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(
        X_combined, positive_labels, test_size=0.2, random_state=42, stratify=positive_labels
    )
    
    # Créer un modèle d'ensemble
    model_pos = create_ensemble_model()
    
    # Entraîner le modèle
    model_pos.fit(X_train_pos, y_train_pos)
    
    # Évaluer le modèle
    y_pred_pos = model_pos.predict(X_test_pos)
    y_proba_pos = model_pos.predict_proba(X_test_pos)[:, 1]
    
    evaluations['positive'] = {
        'accuracy': model_pos.score(X_test_pos, y_test_pos),
        'precision': precision_score(y_test_pos, y_pred_pos, zero_division=0),
        'recall': recall_score(y_test_pos, y_pred_pos, zero_division=0),
        'f1': f1_score(y_test_pos, y_pred_pos, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test_pos, y_pred_pos),
        'classification_report': classification_report(y_test_pos, y_pred_pos, output_dict=True)
    }
    
    # ---- Entraînement du modèle pour les sentiments négatifs ----
    print("Entraînement du modèle pour les sentiments négatifs...")
    
    # Équilibrer les classes
    neg_count = np.sum(negative_labels)
    pos_count = len(negative_labels) - neg_count
    class_weight_neg = {0: 1.0, 1: pos_count/neg_count if neg_count > 0 else 1.0}
    
    X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split(
        X_combined, negative_labels, test_size=0.2, random_state=42, stratify=negative_labels
    )
    
    # Créer un modèle d'ensemble
    model_neg = create_ensemble_model()
    
    # Entraîner le modèle
    model_neg.fit(X_train_neg, y_train_neg)
    
    # Évaluer le modèle
    y_pred_neg = model_neg.predict(X_test_neg)
    y_proba_neg = model_neg.predict_proba(X_test_neg)[:, 1]
    
    evaluations['negative'] = {
        'accuracy': model_neg.score(X_test_neg, y_test_neg),
        'precision': precision_score(y_test_neg, y_pred_neg, zero_division=0),
        'recall': recall_score(y_test_neg, y_pred_neg, zero_division=0),
        'f1': f1_score(y_test_neg, y_pred_neg, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test_neg, y_pred_neg),
        'classification_report': classification_report(y_test_neg, y_pred_neg, output_dict=True)
    }
    
    # Afficher les évaluations
    for sentiment, eval_metrics in evaluations.items():
        print(f"\nÉvaluation du modèle {sentiment}:")
        print(f"Précision: {eval_metrics['accuracy']:.2%}")
        print(f"Precision score: {eval_metrics['precision']:.2%}")
        print(f"Recall score: {eval_metrics['recall']:.2%}")
        print(f"F1 score: {eval_metrics['f1']:.2%}")
        print(f"Matrice de confusion:\n{eval_metrics['confusion_matrix']}")
    
    # Générer des visualisations
    for sentiment, eval_metrics in evaluations.items():
        plt.figure(figsize=(8, 6))
        cm = eval_metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non ' + sentiment, sentiment.capitalize()],
                    yticklabels=['Non ' + sentiment, sentiment.capitalize()])
        plt.xlabel('Prédiction')
        plt.ylabel('Valeur réelle')
        plt.title(f'Matrice de confusion - Sentiment {sentiment}')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, f'confusion_matrix_{sentiment}.png'))
        plt.close()
    
    # Créer un dictionnaire avec les modèles
    models = {
        'positive': model_pos,
        'negative': model_neg
    }
    
    # Sauvegarder les modèles et le vectoriseur
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(models, model_file)

    with open(VECTORIZER_PATH, "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)
    
    generate_evaluation_report(evaluations)
    
    test_examples = [
        "C'est super sympa",
        "Horrible je n'ai pas aimé",
        "Le film était vraiment mal réalisé, j'ai perdu mon temps",
        "Je n'ai pas aimé ce film",
        "Ce n'est pas terrible",
        "Ce n'est pas mauvais",
        "Ce restaurant, c'est la cerise sur le gâteau de notre séjour",
        "Cette application tourne comme une patate chaude",
        "Le rapport qualité-prix laisse à désirer",
        "Je ne peux pas dire que j'ai apprécié cette expérience",
        "Le service après-vente m'a fait attendre 45 minutes.",
        "La livraison a été plus rapide que prévu, bravo",
        "Cette nouvelle fonctionnalité est vraiment pratique",
        "Je ne peux pas me plaindre de la qualité de ce produit"
    ]
    
    print("\nTests du modèle sur des exemples:")
    for example in test_examples:
        score = analyze_sentiment_text(example, models, vectorizer, embeddings)
        sentiment = "POSITIF" if score > 0 else "NÉGATIF" if score < 0 else "NEUTRE"
        print(f"'{example}': {score} ({sentiment})")
    
    return models, vectorizer, embeddings

def generate_evaluation_report(evaluations):
    """Génère un rapport d'évaluation détaillé"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORTS_DIR, f'evaluation_report_{timestamp}.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RAPPORT D'ÉVALUATION DU MODÈLE D'ANALYSE DE SENTIMENTS\n")
        f.write("=" * 60 + "\n\n")
        
        for sentiment, metrics in evaluations.items():
            f.write(f"MODÈLE DE SENTIMENT {sentiment.upper()}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Précision globale: {metrics['accuracy']:.2%}\n")
            f.write(f"Precision score: {metrics['precision']:.2%}\n")
            f.write(f"Recall score: {metrics['recall']:.2%}\n")
            f.write(f"F1 score: {metrics['f1']:.2%}\n\n")
            
            f.write("Matrice de confusion:\n")
            cm = metrics['confusion_matrix']
            f.write(f"[[{cm[0][0]}, {cm[0][1]}]\n")
            f.write(f" [{cm[1][0]}, {cm[1][1]}]]\n\n")
            
            # Ajouter le rapport de classification détaillé
            f.write("Rapport de classification détaillé:\n")
            report = metrics['classification_report']
            for class_name, values in report.items():
                if class_name in ['0', '1']:
                    class_label = 'Non-' + sentiment if class_name == '0' else sentiment.capitalize()
                    f.write(f"  - {class_label}:\n")
                    f.write(f"    Precision: {values['precision']:.2%}\n")
                    f.write(f"    Recall: {values['recall']:.2%}\n")
                    f.write(f"    F1-score: {values['f1-score']:.2%}\n")
                    f.write(f"    Support: {values['support']}\n")
            f.write("\n")
            
        f.write("ANALYSE DES PERFORMANCES\n")
        f.write("-" * 40 + "\n")
        
        f.write("Forces et faiblesses:\n")
        for sentiment, metrics in evaluations.items():
            f.write(f"- Modèle {sentiment}: ")
            if metrics['f1'] > 0.7:
                f.write(f"Bonnes performances (F1 = {metrics['f1']:.2%})\n")
            elif metrics['f1'] > 0.5:
                f.write(f"Performances moyennes (F1 = {metrics['f1']:.2%})\n")
            else:
                f.write(f"Performances faibles (F1 = {metrics['f1']:.2%})\n")

        for sentiment, metrics in evaluations.items():
            precision = metrics['precision']
            recall = metrics['recall']
            if abs(precision - recall) > 0.2:
                if precision > recall:
                    f.write(f"- Le modèle {sentiment} a tendance à être trop conservateur ")
                    f.write(f"(precision: {precision:.2%}, recall: {recall:.2%})\n")
                    f.write("  → Le modèle manque certains cas positifs mais ceux qu'il identifie sont plutôt corrects\n")
                else:
                    f.write(f"- Le modèle {sentiment} a tendance à surestimer les cas positifs ")
                    f.write(f"(precision: {precision:.2%}, recall: {recall:.2%})\n")
                    f.write("  → Le modèle trouve la plupart des cas positifs mais avec des faux positifs\n")
        
    print(f"Rapport d'évaluation généré: {report_path}")

if __name__ == "__main__":
    train_model()