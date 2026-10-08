from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from pathlib import Path

from fastapi import Header, HTTPException

from app.core.config import DATA_DIR, DB_PATH

_SECRET = os.environ.get("CHURNGUARD_AUTH_SECRET", "local-development-secret-change-me").encode()


def init_auth_tables() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at TEXT NOT NULL)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS uploads (
        id TEXT PRIMARY KEY, user_id TEXT NOT NULL, filename TEXT NOT NULL, path TEXT NOT NULL,
        rows INTEGER NOT NULL, columns_json TEXT NOT NULL, created_at TEXT NOT NULL)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS upload_runs (
        id TEXT PRIMARY KEY, upload_id TEXT NOT NULL, user_id TEXT NOT NULL,
        target TEXT NOT NULL, customer_id TEXT, artifact_dir TEXT NOT NULL,
        created_at TEXT NOT NULL)"""
    )
    conn.commit()
    conn.close()


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"{salt.hex()}:{digest.hex()}"


def _check_password(password: str, encoded: str) -> bool:
    salt, expected = encoded.split(":", 1)
    actual = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=2**14, r=8, p=1).hex()
    return hmac.compare_digest(actual, expected)


def _token(user_id: str, email: str) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(
        {"sub": user_id, "email": email, "exp": int(time.time()) + 86400},
        separators=(",", ":"),
    ).encode()).decode().rstrip("=")
    signature = hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}"


def _user_from_token(token: str) -> dict:
    try:
        payload, signature = token.split(".", 1)
        expected = hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError
        data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        if int(data["exp"]) < int(time.time()):
            raise ValueError
        return data
    except (ValueError, KeyError, json.JSONDecodeError, binascii.Error):
        raise HTTPException(401, "Invalid or expired login token")


def register(email: str, password: str) -> dict:
    if len(password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    user_id = secrets.token_hex(12)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("INSERT INTO users VALUES (?, ?, ?, datetime('now'))",
                     (user_id, email.strip().lower(), _hash_password(password)))
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(409, "An account with that email already exists")
    finally:
        conn.close()
    return {"access_token": _token(user_id, email.strip().lower()), "user_id": user_id, "email": email.strip().lower()}


def login(email: str, password: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT id, email, password_hash FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    conn.close()
    if not row or not _check_password(password, row[2]):
        raise HTTPException(401, "Invalid email or password")
    return {"access_token": _token(row[0], row[1]), "user_id": row[0], "email": row[1]}


def current_user(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Login required")
    return _user_from_token(authorization[7:])


def user_upload_dir(user_id: str) -> Path:
    path = DATA_DIR / "uploads" / user_id
    path.mkdir(parents=True, exist_ok=True)
    return path
