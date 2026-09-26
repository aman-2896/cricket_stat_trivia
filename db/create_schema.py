"""
One-off script that creates the Postgres schema (enum types + all tables)
used by the rest of the app. Run directly with `python -m db.create_schema`
to (re)build the database before loading match data with db/loader.py.
"""

from .db import get_connection

def create_enums(conn):
    # Simple/low-complexity: Postgres has no "CREATE TYPE IF NOT EXISTS",
    # so we manually check pg_type first and only create the enum if it's
    # missing, to keep this script safe to re-run.
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 1
        FROM pg_type
        WHERE typname = 'gender';
    """)

    if cursor.fetchone() is None:
        cursor.execute("""
            CREATE TYPE gender AS ENUM ('male', 'female');
        """)

    cursor.execute("""
        SELECT 1
        FROM pg_type
        WHERE typname = 'match_format';
    """)

    if cursor.fetchone() is None:
        cursor.execute("""
            CREATE TYPE match_format AS ENUM ('t20', 'odi', 'test');
        """)

    conn.commit()
    cursor.close()

def create_match_table(conn):
    # Simple/low-complexity: standard CREATE TABLE IF NOT EXISTS, one row
    # per Cricsheet match with its metadata (teams, venue, result, toss...).
    cursor = conn.cursor()
    query = """
        CREATE TABLE IF NOT EXISTS matches(
            id SERIAL PRIMARY KEY,
            cricsheet_id TEXT UNIQUE NOT NULL, 
            city TEXT NOT NULL,
            match_start_date DATE NOT NULL,
            match_end_date DATE,
            event_name TEXT NOT NULL,
            match_number INT,
            match_group TEXT,
            gender gender NOT NULL,
            match_type match_format NOT NULL,
            winner TEXT,
            win_by_runs INT,
            win_by_wickets INT,
            overs INT NOT NULL,
            player_of_match TEXT,
            season_year INT NOT NULL,
            team_type TEXT,
            team_1 TEXT NOT NULL,
            team_2 TEXT NOT NULL,
            toss_winner TEXT,
            toss_winner_decision TEXT,
            venue TEXT NOT NULL
        );
    """

    try:
        cursor.execute(query)
    except Exception as e:
        print(f"Failed to create table 'matches' error {e}")
    else:
        print("=========== Matches table created successfully ===========")

    conn.commit()
    cursor.close()


def create_players_table(conn):
    # Simple/low-complexity: one row per player per match (many-to-one with
    # matches, cascades on delete so removing a match cleans up its players).
    cursor = conn.cursor()

    query = """
        CREATE TABLE IF NOT EXISTS players(
            id SERIAL PRIMARY KEY,
            match_id INT REFERENCES matches(id) ON DELETE CASCADE,
            team TEXT NOT NULL,
            player TEXT NOT NULL
        );
    """

    try:
        cursor.execute(query)
    except Exception as e:
        print(f"Failed to create table 'players' error {e}")
    else:
        print("=========== Players table created successfully ===========")

    conn.commit()
    cursor.close()


def create_innings_table(conn):
    # Simple/low-complexity: one row per innings of a match, tracking
    # powerplay overs and (for chasing innings) the target to reach.
    cursor = conn.cursor()

    query = """
        CREATE TABLE IF NOT EXISTS innings(
            id SERIAL PRIMARY KEY,
            match_id INT REFERENCES matches(id) ON DELETE CASCADE,
            team TEXT NOT NULL,
            powerplay_1_from FLOAT,
            powerplay_1_to FLOAT,
            powerplay_1_type TEXT,
            target_overs INT,
            target_runs INT,
            is_super_over BOOLEAN
        );
    """

    try:
        cursor.execute(query)
    except Exception as e:
        print(f"Failed to create table 'innings' error {e}")
    else:
        print("=========== Innings table created successfully ===========")

    conn.commit()
    cursor.close()


def create_deliveries_table(conn):
    # Core fact table: one row per ball bowled. is_legal_delivery excludes
    # wides/no-balls from ball counts; total_runs vs batter_runs distinguish
    # team/bowler runs conceded from the batter's own runs (see
    # services/text2sql.py SCHEMA_HINTS for how these are used in queries).
    cursor = conn.cursor()

    query = """
        CREATE TABLE IF NOT EXISTS deliveries(
            id SERIAL PRIMARY KEY,
            match_id INT REFERENCES matches(id) ON DELETE CASCADE,
            inning_id INT REFERENCES innings(id) ON DELETE CASCADE,
            over INT NOT NULL,
            is_legal_delivery BOOLEAN NOT NULL,
            delivery_in_over INT NOT NULL,
            batter TEXT NOT NULL,
            bowler TEXT NOT NULL,
            non_striker TEXT NOT NULL,
            batter_runs INT DEFAULT 0,
            extra_runs INT DEFAULT 0,
            total_runs INT DEFAULT 0,
            is_wicket_delivery BOOLEAN NOT NULL,
            is_extra_delivery BOOLEAN NOT NULL,
            extra_type TEXT,
            is_reviewed BOOLEAN NOT NULL
        );
    """

    try:
        cursor.execute(query)
    except Exception as e:
        print(f"Failed to create table 'deliveries' error {e}")
    else:
        print("=========== Deliveries table created successfully ===========")

    conn.commit()
    cursor.close()


def create_wickets_table(conn):
    # Simple/low-complexity: one row per wicket, linked to the delivery that
    # produced it (a delivery can have 0 or occasionally >1 wicket, e.g. run
    # out of non-striker plus original batter in rare cases).
    cursor = conn.cursor()

    query = """
        CREATE TABLE IF NOT EXISTS wickets(
            id SERIAL PRIMARY KEY,
            delivery_id INT REFERENCES deliveries(id)  ON DELETE CASCADE,
            player_out TEXT,
            fielder TEXT,
            wicket_type TEXT
        );
    """

    try:
        cursor.execute(query)
    except Exception as e:
        print(f"Failed to create table 'wickets' error {e}")
    else:
        print("=========== Wickets table created successfully ===========")

    conn.commit()
    cursor.close()


def create_reviews_table(conn):
    # Simple/low-complexity: one row per DRS review taken on a delivery.
    cursor = conn.cursor()

    query = """
        CREATE TABLE IF NOT EXISTS reviews(
            id SERIAL PRIMARY KEY,
            delivery_id INT REFERENCES deliveries(id) ON DELETE CASCADE,
            review_taken_by TEXT,
            review_decision TEXT,
            review_type TEXT
        );
    """

    try:
        cursor.execute(query)
    except Exception as e:
        print(f"Failed to create table 'reviews' error {e}")
    else:
        print("=========== Reviews table created successfully ===========")

    conn.commit()
    cursor.close()


def drop_all_tables(conn):
    """
    Destructive helper (not called by default — see the commented-out call
    at the bottom of this file) that wipes the whole schema so it can be
    rebuilt from scratch during development.
    """
    cursor = conn.cursor()

    tables = [
        "reviews",
        "wickets",
        "deliveries",
        "innings",
        "players",
        "matches"
    ]

    types = [
        "match_format",
        "gender"
    ]

    try:
        # Drop tables in reverse dependency order
        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table};")
            print(f"=========== {table} table dropped successfully ===========")

        # Drop enum types after tables
        for type_name in types:
            cursor.execute(f"DROP TYPE IF EXISTS {type_name};")
            print(f"=========== {type_name} type dropped successfully ===========")

        conn.commit()

    except Exception as e:
        conn.rollback()
        print(f"Failed to drop tables/types: {e}")

    finally:
        cursor.close()


if __name__=="__main__":
    # Order matters: enums must exist before matches (which uses them), and
    # matches must exist before the tables that reference match_id via FK.
    conn=get_connection()
    create_enums(conn)
    create_match_table(conn)
    create_players_table(conn)
    create_innings_table(conn)
    create_deliveries_table(conn)
    create_wickets_table(conn)
    create_reviews_table(conn)

    # drop_all_tables(conn)  # uncomment to wipe the schema before recreating it
    conn.close()
