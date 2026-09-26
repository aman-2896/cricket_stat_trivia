"""
Low-level Postgres connection helpers used by every other module that
needs database access (db/loader.py, services/text2sql.py, services/ingest.py).
"""

import psycopg
from config.config import DB_CONFIG, DB_READONLY_CONFIG


def get_connection():
    """
    Open a full-privilege connection (read + write), used for schema setup,
    data ingestion and vector-store writes.
    """
    try:
        conn = psycopg.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        # Log and re-raise so the caller (and its own try/except) still
        # sees the failure instead of silently getting None back.
        print(f"Unable to connect to database , error {e}")
        raise


def get_readonly_connection():
    """
    Open a connection using the read-only DB role, used specifically for
    executing LLM-generated SQL (services/text2sql.py) so those queries
    cannot modify data even if the safety checks upstream were bypassed.
    """
    try:
        conn = psycopg.connect(**DB_READONLY_CONFIG)
        return conn
    except Exception as e:
        print(f"Unable to connect to database , error {e}")
        raise
