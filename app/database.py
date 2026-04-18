"""
app/database.py
---------------
SQLite database layer.  Two tables:
  users       — authentication (username, hashed password)
  predictions — full prediction history per user
Auto-creates tables on first import.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
import hashlib
from datetime import datetime
from src.config import DB_PATH


# ── Connection helper ────────────────────────────────────────────────────────
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ── Schema ────────────────────────────────────────────────────────────────────
def init_db():
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT    UNIQUE NOT NULL,
            name       TEXT    NOT NULL,
            email      TEXT    UNIQUE NOT NULL,
            password   TEXT    NOT NULL,
            created_at TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            age         REAL,    sex     INTEGER, cp       INTEGER,
            trestbps    REAL,    chol    REAL,    fbs      INTEGER,
            restecg     INTEGER, thalach REAL,    exang    INTEGER,
            oldpeak     REAL,    slope   INTEGER, ca       INTEGER,
            thal        INTEGER,
            model_used  TEXT,
            prediction  INTEGER,
            probability REAL,
            risk_level  TEXT,
            created_at  TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()


# ── Auth ──────────────────────────────────────────────────────────────────────
def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def register_user(username, name, email, password):
    """Returns (True, msg) on success or (False, msg) on failure."""
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO users (username,name,email,password,created_at) VALUES (?,?,?,?,?)",
            (username.strip().lower(), name.strip(),
             email.strip().lower(), _hash(password),
             datetime.now().isoformat()),
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "Username or email already exists."
    finally:
        conn.close()


def login_user(username, password):
    """Returns user dict on success, None on failure."""
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username.strip().lower(), _hash(password)),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ── Predictions ───────────────────────────────────────────────────────────────
def save_prediction(user_id, inputs: dict, model_used: str,
                    prediction: int, probability: float):
    if probability >= 0.65:
        risk = "High Risk"
    elif probability >= 0.35:
        risk = "Moderate Risk"
    else:
        risk = "Low Risk"

    conn = _connect()
    conn.execute("""
        INSERT INTO predictions
          (user_id,age,sex,cp,trestbps,chol,fbs,restecg,thalach,
           exang,oldpeak,slope,ca,thal,model_used,prediction,probability,risk_level,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        user_id,
        inputs.get("age"),      inputs.get("sex"),    inputs.get("cp"),
        inputs.get("trestbps"), inputs.get("chol"),   inputs.get("fbs"),
        inputs.get("restecg"),  inputs.get("thalach"),inputs.get("exang"),
        inputs.get("oldpeak"),  inputs.get("slope"),  inputs.get("ca"),
        inputs.get("thal"),
        model_used, prediction, round(probability, 4), risk,
        datetime.now().isoformat(),
    ))
    conn.commit()
    conn.close()


def get_user_predictions(user_id):
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_predictions():
    conn = _connect()
    rows = conn.execute("""
        SELECT p.*, u.username, u.name
        FROM predictions p
        JOIN users u ON p.user_id = u.id
        ORDER BY p.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Auto-initialise on import
init_db()
