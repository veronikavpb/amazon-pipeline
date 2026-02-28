# src/utils.py
from __future__ import annotations

from urllib.parse import urlparse
import re
import pandas as pd

ASIN_RE = re.compile(r"^[A-Z0-9]{10}$", re.IGNORECASE)


def clean_text(value) -> str:
    """Convert value to a trimmed string (handles NaN safely)."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def is_valid_asin(value: str) -> bool:
    return bool(ASIN_RE.match(value))


def is_valid_url(value: str) -> bool:
    """Basic URL validation (http/https + netloc)."""
    try:
        u = urlparse(value)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def parse_money(value) -> float | None:
    """
    Parses strings like '₹1,099' into 1099.0
    Returns None if not parseable.
    """
    s = clean_text(value)
    if not s:
        return None

    s = s.replace("₹", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_percent(value) -> float | None:
    """Parses strings like '64%' into 64.0"""
    s = clean_text(value)
    if not s:
        return None

    s = s.replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_int(value) -> int | None:
    """Parses '1,234' into 1234"""
    s = clean_text(value)
    if not s:
        return None

    s = s.replace(",", "").strip()
    try:
        return int(float(s))
    except ValueError:
        return None