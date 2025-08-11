import os
import psycopg2
from psycopg2.extras import DictCursor

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    conn = psycopg2.connect(
        host='db',
        dbname=os.environ['POSTGRES_DB'],
        user=os.environ['POSTGRES_USER'],
        password=os.environ['POSTGRES_PASSWORD']
    )
    return conn

def init_db():
    """Initializes the database table if it doesn't exist."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            doc_type VARCHAR(50) NOT NULL,
            extracted_data JSONB,
            original_images BYTEA[],
            face_image BYTEA,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def save_processed_document(doc_type, extracted_data, original_images, face_image):
    """Saves a processed document to the database."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO documents (doc_type, extracted_data, original_images, face_image)
        VALUES (%s, %s, %s, %s) RETURNING id;
        """,
        (doc_type, extracted_data, original_images, face_image)
    )
    new_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return new_id

def get_processed_document(doc_id):
    """Retrieves a processed document from the database by its ID."""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=DictCursor)
    cur.execute("SELECT * FROM documents WHERE id = %s;", (doc_id,))
    document = cur.fetchone()
    cur.close()
    conn.close()
    return document