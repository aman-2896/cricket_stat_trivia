"""
ETL script that reads Cricsheet ball-by-ball JSON files
(https://cricsheet.org format) and loads them into the Postgres schema
defined in db/create_schema.py: matches -> players/innings -> deliveries
-> wickets/reviews.

Run directly with `python -m db.loader` to load every *.json file found in
the ./cricsheets directory.
"""

import json
from pathlib import Path
from .db import get_connection

def load_cricsheet_json(file_path):
    # Simple/low-complexity: just reads and parses one Cricsheet match file.
    with open(file_path,"r",encoding="utf-8") as file:
        return json.load(file)



def normalize_match_type(match_type):
    # Simple/low-complexity: Cricsheet uses mixed-case match type labels
    # ("T20", "ODI", "Test") but our `match_format` enum (db/create_schema.py)
    # is lowercase, so this just maps one to the other.
    mapping={
        "T20":"t20",
        "ODI":"odi",
        "Test":"test",
    }

    return mapping.get(match_type)


def load_match_file(conn, file_path):
    """
    Load one complete Cricsheet JSON match into the database.

    Orchestrates the full pipeline for a single match file: insert the match
    row, then its players, innings and deliveries (which in turn insert any
    wickets/reviews). Skips files that were already loaded (matched by
    cricsheet_id) and rolls back the whole match on any failure so we never
    leave partial data behind.
    """

    file_path = Path(file_path)

    data = load_cricsheet_json(file_path)

    # Use filename as Cricsheet ID.
    # Example:
    # 1234567.json -> 1234567
    #.stem gives filename without extension , property of pathlib.Path
    cricsheet_id = file_path.stem
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT id
            FROM matches
            WHERE cricsheet_id = %s;
            """,
            (cricsheet_id,),
        )

        existing_match = cursor.fetchone()

        if existing_match:
            print(
                f"=========== Match {cricsheet_id} already loaded. Skipping. ==========="
            )
            return existing_match[0]

    finally:
        cursor.close()


    try:
        match_id = load_match(
            conn,
            cricsheet_id,
            data,
        )

        info = data["info"]

        load_players(
            conn,
            match_id,
            info,
        )

        innings_ids = load_innings(
            conn,
            match_id,
            data.get("innings", []),
        )

        load_deliveries(
            conn,
            match_id,
            data.get("innings", []),
            innings_ids,
        )

        conn.commit()

        print(
            f"=========== Match {cricsheet_id} loaded successfully ==========="
        )

        return match_id

    except Exception as e:

        conn.rollback()

        print(
            f"Failed to load match {cricsheet_id}: {e}"
        )

        raise


def load_match(conn, cricsheet_id, data):
    """
    Insert match metadata and return database match ID.
    """

    info = data["info"]

    dates = info["dates"]

    match_start_date = dates[0]
    match_end_date = dates[-1] if len(dates) > 1 else None

    event = info.get("event", {})
    outcome = info.get("outcome", {})
    by = outcome.get("by", {})
    toss = info.get("toss", {})

    player_of_match = info.get("player_of_match", [])

    if player_of_match:
        player_of_match = ", ".join(player_of_match)
    else:
        player_of_match = None

    teams = info["teams"]

    query = """
        INSERT INTO matches (
            cricsheet_id,
            city,
            match_start_date,
            match_end_date,
            event_name,
            match_number,
            match_group,
            gender,
            match_type,
            winner,
            win_by_runs,
            win_by_wickets,
            overs,
            player_of_match,
            season_year,
            team_type,
            team_1,
            team_2,
            toss_winner,
            toss_winner_decision,
            venue
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        RETURNING id;
    """

    values = (
        cricsheet_id,
        info.get("city"),
        match_start_date,
        match_end_date,
        event.get("name"),
        event.get("match_number"),
        event.get("group"),
        info.get("gender"),
        normalize_match_type(info.get("match_type")),
        outcome.get("winner"),
        by.get("runs"),
        by.get("wickets"),
        info.get("overs"),
        player_of_match,
        int(info.get("season")),
        info.get("team_type"),
        teams[0],
        teams[1],
        toss.get("winner"),
        toss.get("decision"),
        info.get("venue"),
    )

    cursor = conn.cursor()

    try:
        cursor.execute(query, values)
        return cursor.fetchone()[0]

    finally:
        cursor.close()


def load_players(conn, match_id, info):
    """
    Insert all players participating in the match.
    """

    query = """
        INSERT INTO players (
            match_id,
            team,
            player
        )
        VALUES (%s, %s, %s);
    """

    cursor = conn.cursor()

    try:
        for team, players in info["players"].items():
            for player in players:
                cursor.execute(
                    query,
                    (
                        match_id,
                        team,
                        player,
                    ),
                )

        print(
            f"=========== Players inserted for match {match_id} ==========="
        )

    except Exception as e:
        print(f"Failed to insert players: {e}")
        raise

    finally:
        cursor.close()


def load_innings(conn, match_id, innings_data):
    query = """
        INSERT INTO innings (
            match_id,
            team,
            powerplay_1_from,
            powerplay_1_to,
            powerplay_1_type,
            target_overs,
            target_runs,
            is_super_over
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s,%s)
        RETURNING id;
    """

    cursor = conn.cursor()

    innings_ids = {}

    try:
        for innings_index, innings in enumerate(innings_data):

            powerplay_from = None
            powerplay_to = None
            powerplay_type = None

            powerplays = innings.get("powerplays", [])
            is_super_over=innings.get("super_over",False)

            if powerplays:
                powerplay = powerplays[0]

                powerplay_from = powerplay.get("from")
                powerplay_to = powerplay.get("to")
                powerplay_type = powerplay.get("type")

            target = innings.get("target", {})

            target_overs = target.get("overs")
            target_runs = target.get("runs")

            cursor.execute(
                query,
                (
                    match_id,
                    innings["team"],
                    powerplay_from,
                    powerplay_to,
                    powerplay_type,
                    target_overs,
                    target_runs,
                    is_super_over
                ),
            )

            innings_id = cursor.fetchone()[0]

            innings_ids[innings_index] = innings_id


        print(
            f"=========== Innings inserted for match {match_id} ==========="
        )

        return innings_ids

    except Exception as e:
        print(f"Failed to insert innings: {e}")
        raise

    finally:
        cursor.close()


def load_deliveries(conn, match_id, innings_data, innings_ids):
    """
    Insert all deliveries, wickets and reviews for a match.

    innings_ids:
        {
            innings_index: database_innings_id
        }
    """

    delivery_query = """
        INSERT INTO deliveries (
            match_id,
            inning_id,
            over,
            is_legal_delivery,
            delivery_in_over,
            batter,
            bowler,
            non_striker,
            batter_runs,
            extra_runs,
            total_runs,
            is_wicket_delivery,
            is_extra_delivery,
            extra_type,
            is_reviewed
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
        RETURNING id;
    """

    wicket_query = """
        INSERT INTO wickets (
            delivery_id,
            player_out,
            fielder,
            wicket_type
        )
        VALUES (%s, %s, %s, %s);
    """

    review_query = """
        INSERT INTO reviews (
            delivery_id,
            review_taken_by,
            review_decision,
            review_type
        )
        VALUES (%s, %s, %s, %s);
    """

    cursor = conn.cursor()

    try:
        for innings_index, innings in enumerate(innings_data):

            innings_id = innings_ids[innings_index]

            for over_data in innings.get("overs", []):

                over_number = over_data["over"]

                for delivery_number, delivery in enumerate(
                    over_data.get("deliveries", []),
                    start=1
                ):

                    runs = delivery.get("runs", {})
                    extras = delivery.get("extras", {})

                    batter_runs = runs.get("batter", 0)
                    extra_runs = runs.get("extras", 0)
                    total_runs = runs.get("total", 0)

                    # -------------------------------------------------
                    # Extras
                    # -------------------------------------------------

                    is_extra_delivery = bool(extras)

                    extra_type = None

                    if extras:
                        # Normally there will be one extra type.
                        # If there are multiple, take the first one.
                        extra_type = next(iter(extras))

                    # -------------------------------------------------
                    # Legal delivery
                    # -------------------------------------------------

                    # Wides and no-balls are illegal deliveries.
                    is_legal_delivery = not (
                        "wides" in extras
                        or "noballs" in extras
                    )

                    # -------------------------------------------------
                    # Wickets
                    # -------------------------------------------------

                    wickets = delivery.get("wickets", [])

                    is_wicket_delivery = bool(wickets)

                    # -------------------------------------------------
                    # Review
                    # -------------------------------------------------

                    review = delivery.get("review")

                    is_reviewed = review is not None

                    # -------------------------------------------------
                    # Delivery number
                    # -------------------------------------------------

                    actual_delivery = delivery.get("actual_delivery")

                    if actual_delivery is not None:
                        # Cricsheet sometimes records the "true" ball number
                        # as "<over>.<ball>" (e.g. "12.3") to account for
                        # re-bowled deliveries; the digits after the dot are
                        # the actual position within the over.
                        delivery_in_over = int(
                            actual_delivery.split(".")[1]
                        )
                    else:
                        # Fallback: no override given, so just use this
                        # delivery's 1-based position in the overs[] list.
                        delivery_in_over = delivery_number

                    # -------------------------------------------------
                    # Insert delivery
                    # -------------------------------------------------

                    cursor.execute(
                        delivery_query,
                        (
                            match_id,
                            innings_id,
                            over_number,
                            is_legal_delivery,
                            delivery_in_over,
                            delivery["batter"],
                            delivery["bowler"],
                            delivery["non_striker"],
                            batter_runs,
                            extra_runs,
                            total_runs,
                            is_wicket_delivery,
                            is_extra_delivery,
                            extra_type,
                            is_reviewed,
                        ),
                    )

                    delivery_id = cursor.fetchone()[0]

                    # -------------------------------------------------
                    # Insert wickets
                    # -------------------------------------------------

                    for wicket in wickets:

                        fielders = wicket.get("fielders", [])

                        # A wicket can theoretically have multiple
                        # fielders involved.
                        fielder_names = [
                            fielder.get("name")
                            for fielder in fielders
                            if fielder.get("name")
                        ]

                        fielder = ", ".join(fielder_names) or None

                        cursor.execute(
                            wicket_query,
                            (
                                delivery_id,
                                wicket.get("player_out"),
                                fielder,
                                wicket.get("kind"),
                            ),
                        )

                    # -------------------------------------------------
                    # Insert review
                    # -------------------------------------------------

                    if review:

                        cursor.execute(
                            review_query,
                            (
                                delivery_id,
                                review.get("by"),
                                review.get("decision"),
                                review.get("type"),
                            ),
                        )

        print(
            f"=========== Deliveries loaded for match {match_id} ==========="
        )

    except Exception as e:
        print(f"Failed to insert deliveries: {e}")
        raise

    finally:
        cursor.close()


if __name__ == "__main__":

    conn = get_connection()

    cricsheet_dir=Path("./cricsheets")
    try:
        for file_path in cricsheet_dir.glob("*.json"):
                load_match_file(
                    conn,
                    file_path,
                )

    finally:
        conn.close()
