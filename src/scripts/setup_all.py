import os
import sys
import logging
from datetime import datetime
import time

# Configuration du logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f'setup_all_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    # Étape 1: Configuration de la base de données
    logging.info("Étape 1/3: Configuration de la base de données...")
    try:
        from src.database.setup_db import setup_database
        setup_database()
        logging.info("Base de données configurée avec succès.")
    except Exception as e:
        logging.error(f"Erreur lors de la configuration de la base de données: {str(e)}")
        return False
    
    # Attendre que la base de données soit prête
    time.sleep(2)
    
    # Étape 2: Génération du dataset
    logging.info("Étape 2/3: Génération du dataset...")
    try:
        from src.scripts.generate_dataset import generate_dataset
        generate_dataset()
        logging.info("Dataset généré avec succès.")
    except Exception as e:
        logging.error(f"Erreur lors de la génération du dataset: {str(e)}")
        return False
    
    # Étape 3: Entraînement du modèle
    logging.info("Étape 3/3: Entraînement du modèle...")
    try:
        from src.scripts.retrain import main as retrain_main
        success = retrain_main()
        if success:
            logging.info("Modèle entraîné avec succès.")
        else:
            logging.error("Échec de l'entraînement du modèle.")
            return False
    except Exception as e:
        logging.error(f"Erreur lors de l'entraînement du modèle: {str(e)}")
        return False
    
    logging.info("Installation complète réussie! L'API est prête à être utilisée.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 