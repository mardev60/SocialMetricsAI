from flask import Flask, request, jsonify, send_file
from src.database.database import insert_tweet, get_all_tweets
import pickle
import os
from src.models.sentiment_model import analyze_sentiment_text
import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend non-interactif

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

@app.route('/generate_report', methods=['GET'])
def generate_report():
    """Génère un rapport PDF contenant la matrice de confusion et d'autres métriques."""
    if models is None or vectorizer is None:
        return jsonify({"error": "le modèle n'est pas chargé"}), 500
    
    # Récupérer les tweets de la base de données
    tweets = get_all_tweets()
    
    # Créer un buffer pour le PDF
    buffer = io.BytesIO()
    
    # Créer le document PDF
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Ajouter le titre
    title = Paragraph("Rapport d'analyse de sentiments", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Ajouter une introduction
    intro = Paragraph("Ce rapport présente les résultats de l'analyse de sentiments sur les tweets stockés dans la base de données.", styles['Normal'])
    story.append(intro)
    story.append(Spacer(1, 12))
    
    # Générer des statistiques sur les tweets
    if tweets:
        # Compter les tweets positifs et négatifs
        positive_count = sum(1 for tweet in tweets if tweet['positive'] > tweet['negative'])
        negative_count = len(tweets) - positive_count
        
        # Créer un graphique à barres pour les sentiments
        plt.figure(figsize=(6, 4))
        plt.bar(['Positif', 'Négatif'], [positive_count, negative_count], color=['green', 'red'])
        plt.title('Répartition des sentiments')
        plt.ylabel('Nombre de tweets')
        
        # Sauvegarder le graphique dans un buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plt.close()
        
        # Ajouter l'image au PDF
        img = Image(img_buffer, width=400, height=300)
        story.append(img)
        story.append(Spacer(1, 12))
        
        # Créer une matrice de confusion simple
        confusion_matrix = [
            ['', 'Prédit Positif', 'Prédit Négatif'],
            ['Réel Positif', positive_count, 0],
            ['Réel Négatif', 0, negative_count]
        ]
        
        # Ajouter un titre pour la matrice
        matrix_title = Paragraph("Matrice de confusion (simplifiée)", styles['Heading2'])
        story.append(matrix_title)
        story.append(Spacer(1, 6))
        
        # Créer une table pour la matrice de confusion
        matrix_table = Table(confusion_matrix)
        matrix_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), colors.white),
            ('BACKGROUND', (1, 0), (2, 0), colors.lightgrey),
            ('BACKGROUND', (0, 1), (0, 2), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(matrix_table)
        story.append(Spacer(1, 12))
        
        # Ajouter quelques exemples de tweets avec leur sentiment
        examples_title = Paragraph("Exemples de tweets analysés", styles['Heading2'])
        story.append(examples_title)
        story.append(Spacer(1, 6))
        
        # Limiter à 5 exemples maximum
        example_tweets = tweets[:5]
        
        # Créer un tableau pour les exemples
        example_data = [['Tweet', 'Score Positif', 'Score Négatif', 'Sentiment']]
        for tweet in example_tweets:
            sentiment = "Positif" if tweet['positive'] > tweet['negative'] else "Négatif"
            example_data.append([
                tweet['text'][:100] + ('...' if len(tweet['text']) > 100 else ''),
                f"{tweet['positive']:.4f}",
                f"{tweet['negative']:.4f}",
                sentiment
            ])
        
        example_table = Table(example_data, colWidths=[250, 70, 70, 70])
        example_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(example_table)
    else:
        no_data = Paragraph("Aucune donnée disponible dans la base de données.", styles['Normal'])
        story.append(no_data)
    
    # Construire le PDF
    doc.build(story)
    
    # Préparer le buffer pour l'envoi
    buffer.seek(0)
    
    # Créer le répertoire reports s'il n'existe pas
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Sauvegarder le PDF sur le disque
    pdf_path = os.path.join(reports_dir, "rapport_sentiment_analysis.pdf")
    with open(pdf_path, 'wb') as f:
        f.write(buffer.getvalue())
    
    # Retourner le PDF
    return send_file(
        io.BytesIO(buffer.getvalue()),
        mimetype='application/pdf',
        as_attachment=True,
        download_name='rapport_sentiment_analysis.pdf'
    )

if __name__ == '__main__':
    app.run(debug=True)