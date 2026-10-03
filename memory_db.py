import sqlite3
import os


DB_PATH = os.path.join(os.path.dirname(__file__), "campus_drone.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS bots (
            bot_id TEXT PRIMARY KEY,
            battery_pct REAL,
            location TEXT,
            status TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id TEXT,
            pickup TEXT,
            dropoff TEXT,
            floor INTEGER,
            item TEXT,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # seed two drones if table is empty
    c.execute("SELECT COUNT(*) FROM bots")
    if c.fetchone()[0] == 0:
        c.executemany("INSERT INTO bots VALUES (?, ?, ?, ?)", [
            ("Drone-1", 85.0, "Dock", "idle"),
            ("Drone-2", 62.0, "Dock", "idle"),
        ])

    conn.commit()
    conn.close()

def get_bots():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM bots").fetchall()
    conn.close()
    return [{"bot_id": r[0], "battery_pct": r[1], "location": r[2], "status": r[3]} for r in rows]

# so the dashboard can save battery and location after a flight
def update_bot(bot_id, battery_pct, location, status):
    conn = get_connection()
    conn.execute(
        "UPDATE bots SET battery_pct = ?, location = ?, status = ? WHERE bot_id = ?",
        (battery_pct, location, status, bot_id)
    )
    conn.commit()
    conn.close()

#  status can be passed in now (delivered / rejected)
def add_task(bot_id, pickup, dropoff, floor, item, status="pending"):
    conn = get_connection()
    conn.execute(
        "INSERT INTO tasks (bot_id, pickup, dropoff, floor, item, status) VALUES (?, ?, ?, ?, ?, ?)",
        (bot_id, pickup, dropoff, floor, item, status)
    )
    conn.commit()
    conn.close()

def get_tasks():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM tasks ORDER BY created_at DESC LIMIT 10").fetchall()
    conn.close()
    return [{"id": r[0], "drone": r[1], "pickup": r[2], "dropoff": r[3], "floor": r[4], "item": r[5], "status": r[6]} for r in rows]