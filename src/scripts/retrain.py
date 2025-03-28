import os
import sys
import logging
from datetime import datetime
from src.models.sentiment_model import train_model

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
        return True
    
    except Exception as e:
        logging.error(f"erreur lors du reentrainement du modele: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 