import mysql.connector
from config import DB_CONFIG

def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"Erreur de connexion à MySQL : {err}")
        return None

if __name__ == "__main__":
    conn = get_db_connection()
    if conn:
        print("Connexion réussie à MySQL")
        conn.close()