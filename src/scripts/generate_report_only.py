#!/usr/bin/env python
"""
Script pour générer uniquement un rapport PDF d'analyse de sentiments.
Ce script utilise le modèle déjà entraîné pour créer un rapport PDF.
"""
import os
import sys
import logging
from datetime import datetime

# Ajouter le répertoire parent au chemin pour permettre d'importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.scripts.generate_report import generate_pdf_report

def main():
    """
    Génère un rapport PDF d'analyse de sentiments sans réentraîner le modèle.
    
    Returns:
        bool: True si la génération du rapport a réussi, False sinon.
    """
    # Configuration du logging
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f'generate_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("Génération du rapport PDF...")
    
    try:
        # Créer le répertoire reports s'il n'existe pas
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Définir le nom du fichier avec un timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Permettre de spécifier un chemin de sortie en argument
        output_path = None
        if len(sys.argv) > 1:
            output_path = sys.argv[1]
        else:
            output_path = os.path.join(reports_dir, f"rapport_sentiment_analysis_{timestamp}.pdf")
        
        # Générer le rapport PDF
        report_path = generate_pdf_report(output_path)
        logging.info(f"Rapport PDF généré avec succès: {report_path}")
        
        return True
    
    except Exception as e:
        logging.error(f"Erreur lors de la génération du rapport PDF: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 