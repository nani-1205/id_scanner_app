import os
import psycopg2
from psycopg2.extras import DictCursor

def get_db_connection():
    """
    Establishes a connection to the PostgreSQL database.
    """
    conn = psycopg2.connect(
        # <<< THIS IS THE CORRECT CONFIGURATION >>>
        # Using the Docker service name 'db' as the host allows containers on the same
        # custom bridge network to resolve and connect to each other.
        host='db',
        port='5432', # It's good practice to be explicit with the default port
        dbname=os.environ['POSTGRES_DB'],
        user=os.environ['POSTGRES_USER'],
        password=os.environ['POSTGRES_PASSWORD']
    )
    return conn

def init_db():
    """Initializes the database table if it doesn't exist."""
    conn = get_db_connection()
    cur = conn.cursor()
    # This SQL creates the 'documents' table with columns for all our data.
    # JSONB is a highly efficient format for storing the extracted JSON data.
    # BYTEA[] allows storing an array of binary data (our original images).
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
    # Get the ID of the newly created row to return to the application
    new_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return new_id

def get_processed_document(doc_id):
    """Retrieves a single processed document from the database by its ID."""
    conn = get_db_connection()
    # DictCursor allows us to access columns by name (e.g., document['extracted_data'])
    cur = conn.cursor(cursor_factory=DictCursor)
    cur.execute("SELECT * FROM documents WHERE id = %s;", (doc_id,))
    document = cur.fetchone()
    cur.close()
    conn.close()
    return document

def get_history(page=1, per_page=10):
    """
    Retrieves a paginated list of processed documents for the history page.
    """
    offset = (page - 1) * per_page
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=DictCursor)
    
    # First, get the total count of all documents for pagination logic.
    cur.execute("SELECT COUNT(*) FROM documents;")
    total_count = cur.fetchone()[0]
    
    # Then, fetch only the specific page of results, ordered by most recent.
    cur.execute(
        "SELECT id, doc_type, created_at FROM documents ORDER BY created_at DESC LIMIT %s OFFSET %s;",
        (per_page, offset)
    )
    history = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return history, total_count