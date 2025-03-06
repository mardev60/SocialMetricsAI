from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pickle
from db import get_all_tweets

def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return model, {
        'accuracy': model.score(X_test, y_test),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

def train_model():
    tweets = get_all_tweets()
    if not tweets:
        print("Aucune donnée disponible pour l'entraînement.")
        return None, None

    texts = [tweet["text"] for tweet in tweets]
    labels = {
        "positive": [tweet["positive"] for tweet in tweets],
        "negative": [tweet["negative"] for tweet in tweets]
    }

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)

    models = {}
    evaluations = {}

    for sentiment, y in labels.items():
        models[sentiment], evaluations[sentiment] = train_and_evaluate_model(X, y)

    with open("sentiment_model.pkl", "wb") as model_file:
        pickle.dump(models, model_file)

    with open("vectorizer.pkl", "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)

    return models, vectorizer

if __name__ == "__main__":
    train_model()