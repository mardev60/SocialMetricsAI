import mysql.connector
from src.database.config import DB_CONFIG

def setup_database():
    conn = mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"]
    )
    
    cursor = conn.cursor()
    
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
    print(f"Base de données '{DB_CONFIG['database']}' créée ou déjà existante.")
    
    cursor.execute(f"USE {DB_CONFIG['database']}")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tweets (
        id INT AUTO_INCREMENT PRIMARY KEY,
        text TEXT NOT NULL,
        positive TINYINT NOT NULL,
        negative TINYINT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    print("Table 'tweets' créée ou déjà existante.")
    
    sample_tweets = [
        ("J'adore ce nouveau produit, il est fantastique !", 1, 0),
        ("Ce service est excellent, je le recommande vivement.", 1, 0),
        ("Quelle journée magnifique, je suis très heureux.", 1, 0),
        ("Je déteste ce service, c'est vraiment terrible.", 0, 1),
        ("Ce produit est de mauvaise qualité, je suis déçu.", 0, 1),
        ("Expérience horrible, je ne reviendrai jamais.", 0, 1),
        ("Le film était correct, ni bon ni mauvais.", 0, 0),
        ("Je ne sais pas quoi penser de cette situation.", 0, 0)
    ]
    
    try:
        cursor.executemany(
            "INSERT INTO tweets (text, positive, negative) VALUES (%s, %s, %s)",
            sample_tweets
        )
        conn.commit()
        print(f"{cursor.rowcount} tweets d'exemple insérés.")
    except mysql.connector.Error as err:
        if err.errno == 1062:
            print("Les tweets d'exemple existent déjà.")
        else:
            print(f"Erreur lors de l'insertion des tweets d'exemple : {err}")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    setup_database() 