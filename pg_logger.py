import psycopg2
import datetime

class PGLogger:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname="agent",
            user="postgres",
            password="password",
            host="localhost"
        )

    def log(self, session_id, user_message, agent_response):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO chat_logs(session_id, user_message, agent_response, created_at)
            VALUES (%s, %s, %s, %s)
        """, (
            session_id,
            user_message,
            agent_response,
            datetime.datetime.now()
        ))
        self.conn.commit()