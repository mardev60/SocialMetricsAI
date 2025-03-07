import mysql.connector
from src.database.config import DB_CONFIG

def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"Erreur de connexion à MySQL : {err}")
        return None

def insert_tweet(text, positive, negative):
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO tweets (text, positive, negative) VALUES (%s, %s, %s)",
                (text, positive, negative)
            )
            conn.commit()
            cursor.close()
        except mysql.connector.Error as err:
            print(f"Erreur lors de l'insertion du tweet : {err}")
        finally:
            conn.close()

def get_all_tweets():
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM tweets")
            tweets = cursor.fetchall()
            cursor.close()
            return tweets
        except mysql.connector.Error as err:
            print(f"Erreur lors de la récupération des tweets : {err}")
            return []
        finally:
            conn.close()