from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from src.database import connect, utc_now


PASSWORD_MIN_LENGTH = 12
_SCRYPT_N = 1 << 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_ATTEMPT_WINDOW_SECONDS = 15 * 60
_LOCKOUT_SECONDS = 15 * 60
_MAX_LOGIN_FAILURES = 5
_DUMMY_PASSWORD_HASH = ""


class AuthenticationError(ValueError):
    pass


def normalize_email(email: str) -> str:
    value = email.strip().casefold()
    if len(value) > 254 or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
        raise ValueError("Enter a valid email address.")
    return value


def hash_password(password: str) -> str:
    if len(password) < PASSWORD_MIN_LENGTH:
        raise ValueError(
            f"Password must contain at least {PASSWORD_MIN_LENGTH} characters."
        )
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=64,
    )
    return "$".join(
        (
            "scrypt",
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            base64.urlsafe_b64encode(salt).decode("ascii"),
            base64.urlsafe_b64encode(digest).decode("ascii"),
        )
    )


def _dummy_password_hash() -> str:
    global _DUMMY_PASSWORD_HASH
    if not _DUMMY_PASSWORD_HASH:
        _DUMMY_PASSWORD_HASH = hash_password(secrets.token_urlsafe(32))
    return _DUMMY_PASSWORD_HASH


def verify_password(password: str, encoded_hash: str) -> bool:
    try:
        algorithm, n_value, r_value, p_value, salt_text, digest_text = (
            encoded_hash.split("$")
        )
        if algorithm != "scrypt":
            return False
        n_value, r_value, p_value = (
            int(n_value),
            int(r_value),
            int(p_value),
        )
        if (
            n_value < 2
            or n_value > 1 << 18
            or r_value < 1
            or r_value > 32
            or p_value < 1
            or p_value > 16
        ):
            return False
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_text.encode("ascii"))
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n_value,
            r=r_value,
            p=p_value,
            dklen=len(expected),
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError, UnicodeError, MemoryError):
        return False


def register_user(name: str, email: str, password: str) -> dict[str, str]:
    display_name = " ".join(name.split())
    if not display_name or len(display_name) > 120:
        raise ValueError("Enter a name between 1 and 120 characters.")
    email_key = normalize_email(email)
    password_hash = hash_password(password)
    user_id = str(uuid.uuid4())
    created_at = utc_now()
    try:
        with connect() as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO users (id, name, email, password_hash, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_id, display_name, email_key, password_hash, created_at),
                )
                connection.execute(
                    """
                    INSERT INTO user_settings
                        (user_id, setting_key, setting_value, updated_at)
                    VALUES (?, 'currency', 'INR', ?)
                    """,
                    (user_id, created_at),
                )
    except sqlite3.IntegrityError as error:
        if "users.email" in str(error).casefold() or "unique" in str(error).casefold():
            raise ValueError("An account with that email already exists.") from None
        raise
    return {"id": user_id, "name": display_name, "email": email_key}


def _session_ttl_minutes() -> int:
    try:
        value = int(os.environ.get("FINTWIN_SESSION_TTL_MINUTES", "480"))
    except ValueError as error:
        raise RuntimeError("FINTWIN_SESSION_TTL_MINUTES must be an integer.") from error
    if not 5 <= value <= 10080:
        raise RuntimeError(
            "FINTWIN_SESSION_TTL_MINUTES must be between 5 and 10080."
        )
    return value


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(user_id: str) -> tuple[str, datetime]:
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=_session_ttl_minutes())
    with connect() as connection:
        with connection:
            user = connection.execute(
                "SELECT id FROM users WHERE id = ? AND is_active = 1", (user_id,)
            ).fetchone()
            if user is None:
                raise AuthenticationError("The account is not available.")
            connection.execute(
                """
                INSERT INTO auth_sessions (token_hash, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (_token_digest(token), user_id, now.isoformat(), expires.isoformat()),
            )
    return token, expires


def _record_failed_login(email_key: str, now: float) -> bool:
    with connect() as connection:
        with connection:
            row = connection.execute(
                "SELECT window_started, failures, locked_until "
                "FROM auth_attempts WHERE email_key = ?",
                (email_key,),
            ).fetchone()
            if row is None or now - float(row["window_started"]) >= _ATTEMPT_WINDOW_SECONDS:
                connection.execute(
                    """
                    INSERT INTO auth_attempts
                        (email_key, window_started, failures, locked_until)
                    VALUES (?, ?, 1, 0)
                    ON CONFLICT(email_key) DO UPDATE SET
                        window_started=excluded.window_started,
                        failures=1,
                        locked_until=0
                    """,
                    (email_key, now),
                )
                return False
            failures = int(row["failures"]) + 1
            locked_until = (
                now + _LOCKOUT_SECONDS if failures >= _MAX_LOGIN_FAILURES else 0
            )
            connection.execute(
                """
                UPDATE auth_attempts
                SET failures = ?, locked_until = ?
                WHERE email_key = ?
                """,
                (failures, locked_until, email_key),
            )
            return locked_until > 0


def authenticate(email: str, password: str) -> tuple[dict[str, str], str, datetime]:
    try:
        email_key = normalize_email(email)
    except ValueError:
        email_key = "invalid:" + hashlib.sha256(email.strip().casefold().encode()).hexdigest()
    now = time.time()
    with connect() as connection:
        row = connection.execute(
            """
            SELECT id, name, email, password_hash, is_active
            FROM users WHERE email = ? COLLATE NOCASE
            """,
            (email_key,),
        ).fetchone()
        attempt = connection.execute(
            "SELECT locked_until FROM auth_attempts WHERE email_key = ?",
            (email_key,),
        ).fetchone()
    if attempt is not None and float(attempt["locked_until"]) > now:
        raise AuthenticationError(
            "Too many sign-in attempts. Wait a little before trying again."
        )
    password_hash = (
        str(row["password_hash"])
        if row is not None and row["is_active"]
        else _dummy_password_hash()
    )
    password_is_valid = verify_password(password, password_hash)
    if row is None or not row["is_active"] or not password_is_valid:
        locked = _record_failed_login(email_key, now)
        if locked:
            raise AuthenticationError(
                "Too many sign-in attempts. Wait a little before trying again."
            )
        raise AuthenticationError("Email or password is incorrect.")
    with connect() as connection:
        with connection:
            connection.execute(
                "DELETE FROM auth_attempts WHERE email_key = ?", (email_key,)
            )
    user = {"id": str(row["id"]), "name": str(row["name"]), "email": str(row["email"])}
    token, expires = create_session(user["id"])
    return user, token, expires


def get_session_user(token: str) -> dict[str, str] | None:
    if not token or len(token) > 256:
        return None
    now = datetime.now(timezone.utc).isoformat()
    with connect() as connection:
        row = connection.execute(
            """
            SELECT u.id, u.name, u.email
            FROM auth_sessions AS s
            JOIN users AS u ON u.id = s.user_id
            WHERE s.token_hash = ? AND s.revoked_at IS NULL
              AND s.expires_at > ? AND u.is_active = 1
            """,
            (_token_digest(token), now),
        ).fetchone()
    if row is None:
        return None
    return {"id": str(row["id"]), "name": str(row["name"]), "email": str(row["email"])}


def revoke_session(token: str) -> None:
    if not token:
        return
    with connect() as connection:
        with connection:
            connection.execute(
                """
                UPDATE auth_sessions SET revoked_at = ?
                WHERE token_hash = ? AND revoked_at IS NULL
                """,
                (utc_now(), _token_digest(token)),
            )
