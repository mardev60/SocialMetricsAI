import os
import sys
import logging
from datetime import datetime
from src.models.sentiment_model import train_model
from src.scripts.generate_report import generate_pdf_report

log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f'retrain_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    logging.info("debut du reentrainement du modele...")
    
    try:
        models, vectorizer, embeddings = train_model()
        
        if models is None or vectorizer is None:
            logging.error("echec de l'entrainement du modele, aucune donnee disponible")
            return False
        
        logging.info("modele reentraine avec succes")
        
        # Générer automatiquement le rapport PDF
        logging.info("Generation du rapport PDF...")
        try:
            reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            # Définir le nom du fichier avec un timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_path = os.path.join(reports_dir, f"rapport_sentiment_analysis_{timestamp}.pdf")
            
            # Générer le rapport PDF
            report_path = generate_pdf_report(pdf_path)
            logging.info(f"Rapport PDF généré avec succès: {report_path}")
        except Exception as e:
            logging.error(f"Erreur lors de la génération du rapport PDF: {str(e)}")
            # Ne pas faire échouer l'entraînement si la génération du rapport échoue
        
        return True
    
    except Exception as e:
        logging.error(f"erreur lors du reentrainement du modele: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 