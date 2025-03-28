#!/usr/bin/env python
"""
Script pour générer un rapport PDF d'analyse de sentiments.
Ce script crée un PDF contenant une matrice de confusion et d'autres métriques.
"""
import os
import sys
import io
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend non-interactif

# Ajouter le répertoire parent au chemin pour permettre d'importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from src.database.database import get_all_tweets

def generate_pdf_report(output_path=None):
    """
    Génère un rapport PDF contenant la matrice de confusion et d'autres métriques.
    
    Args:
        output_path (str, optional): Chemin où sauvegarder le PDF. 
                                     Si None, le fichier sera sauvegardé dans le répertoire reports.
    
    Returns:
        str: Le chemin vers le fichier PDF généré.
    """
    # Définir le chemin de sortie par défaut si non spécifié
    if output_path is None:
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        output_path = os.path.join(reports_dir, "rapport_sentiment_analysis.pdf")
    
    # Récupérer les tweets de la base de données
    tweets = get_all_tweets()
    
    # Utiliser des données d'exemple si aucun tweet n'est disponible
    use_example_data = len(tweets) == 0
    if use_example_data:
        tweets = [
            {"id": 1, "text": "J'adore ce produit, il est fantastique!", "positive": 0.92, "negative": 0.08},
            {"id": 2, "text": "Le service client est très réactif et professionnel.", "positive": 0.87, "negative": 0.13},
            {"id": 3, "text": "Expérience décevante, je ne recommande pas.", "positive": 0.15, "negative": 0.85},
            {"id": 4, "text": "Produit de qualité moyenne, peut mieux faire.", "positive": 0.45, "negative": 0.55},
            {"id": 5, "text": "Super expérience, je suis totalement satisfait!", "positive": 0.95, "negative": 0.05},
            {"id": 6, "text": "Livraison en retard et produit endommagé.", "positive": 0.08, "negative": 0.92},
            {"id": 7, "text": "Rapport qualité-prix imbattable.", "positive": 0.89, "negative": 0.11},
            {"id": 8, "text": "Interface utilisateur complexe et difficile à utiliser.", "positive": 0.20, "negative": 0.80}
        ]
    
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
    intro_text = "Ce rapport présente les résultats de l'analyse de sentiments sur les tweets stockés dans la base de données."
    if use_example_data:
        intro_text += " <i>Note: Ce rapport utilise des données d'exemple car aucun tweet n'a été trouvé dans la base de données.</i>"
    intro = Paragraph(intro_text, styles['Normal'])
    story.append(intro)
    story.append(Spacer(1, 12))
    
    # Générer des statistiques sur les tweets
    if tweets:
        # Compter les tweets positifs et négatifs
        positive_count = sum(1 for tweet in tweets if tweet['positive'] > tweet['negative'])
        negative_count = len(tweets) - positive_count
        
        # Ajouter un résumé des statistiques
        stats_text = f"Nombre total de tweets analysés: {len(tweets)}<br/>"
        stats_text += f"Tweets positifs: {positive_count} ({positive_count/len(tweets)*100:.1f}%)<br/>"
        stats_text += f"Tweets négatifs: {negative_count} ({negative_count/len(tweets)*100:.1f}%)"
        stats = Paragraph(stats_text, styles['Normal'])
        story.append(stats)
        story.append(Spacer(1, 12))
        
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
        # Pour une vraie matrice de confusion, nous aurions besoin des valeurs réelles (ground truth)
        # Ici, on simplifie en supposant que tous les tweets sont correctement classifiés
        confusion_matrix = [
            ['', 'Prédit Positif', 'Prédit Négatif'],
            ['Réel Positif', positive_count, 0],
            ['Réel Négatif', 0, negative_count]
        ]
        
        # Ajouter un titre pour la matrice
        matrix_title = Paragraph("Matrice de confusion (simplifiée)", styles['Heading2'])
        story.append(matrix_title)
        story.append(Spacer(1, 6))
        
        # Ajouter une explication de la matrice
        matrix_explanation = Paragraph(
            "Cette matrice de confusion est simplifiée car nous ne disposons pas des véritables étiquettes pour tous les tweets. "
            "Elle suppose que tous les tweets ont été correctement classifiés par le modèle. Dans un cas réel, "
            "les valeurs hors diagonale représenteraient les erreurs de classification.",
            styles['Normal']
        )
        story.append(matrix_explanation)
        story.append(Spacer(1, 12))
        
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
    
    # Sauvegarder le PDF sur le disque
    with open(output_path, 'wb') as f:
        f.write(buffer.getvalue())
    
    print(f"Rapport PDF généré avec succès : {output_path}")
    return output_path

if __name__ == "__main__":
    # Permettre de spécifier un chemin de sortie en argument
    output_path = None
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    
    generate_pdf_report(output_path) 