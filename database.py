import sqlite3
from datetime import datetime
DB_FILE="plammind.db"

def init_db()->None:
    """Initializes SQLite tables for document metadata and interview bookings."""
    conn=sqlite3.connect(DB_FILE)
    cursor=conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            strategy TEXT NOT NULL,
            total_chunks INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interview_bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

def save_document_metadata(filename:str, strategy:str,total_chunks:int)->None:
    conn=sqlite3.connect(DB_FILE)
    cursor=conn.cursor()
    cursor.execute(
        "INSERT INTO documents (filename, strategy, total_chunks, created_at) VALUES (?, ?, ?, ?)",
        (filename, strategy, total_chunks, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

def save_interview_booking(session_id:str, name:str, email:str,date:str, time:str)->None:
    conn=sqlite3.connect(DB_FILE)
    cursor=conn.cursor()
    cursor.execute(
        "INSERT INTO interview_bookings (session_id, name, email, date, time, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, name, email, date, time, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    

    


