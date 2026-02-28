# src/validator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any
from urllib.parse import urlparse
import re
import pandas as pd


ASIN_RE = re.compile(r"^[A-Z0-9]{10}$", re.IGNORECASE)


@dataclass
class ValidationIssue:
    row_index: int | None          # None means "file-level" error (e.g., missing columns)
    column: str | None
    rule: str
    value: Any
    message: str

    def to_dict(self) -> dict:
        return {
            "row_index": self.row_index,
            "column": self.column,
            "rule": self.rule,
            "value": self.value,
            "message": self.message,
        }


def _is_valid_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def _parse_money(value: str) -> float | None:
    """
    Parses currency strings like: '₹1,099' into 1099.0
    Returns None if not parseable.
    """
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None

    # Remove currency symbol and commas and whitespace
    s = s.replace("₹", "").replace(",", "").strip()

    try:
        return float(s)
    except ValueError:
        return None


def _parse_percent(value: str) -> float | None:
    """
    Parses percent strings like: '64%' into 64.0
    """
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None

    s = s.replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _parse_int(value: str) -> int | None:
    """
    Parses strings like: '1,234' into 1234
    """
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None

    s = s.replace(",", "").strip()
    try:
        return int(float(s))
    except ValueError:
        return None


@dataclass
class Validator:
    """
    Validator returns a list of issues (instead of raising immediately).
    That's useful because:
    - you can log ALL problems at once into an error log
    - you can choose thresholds (e.g., allow 2% bad rows) later if needed
    """
    required_columns: list[str]

    def validate(self, df: pd.DataFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # ---- 1) Schema validation ----
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            issues.append(
                ValidationIssue(
                    row_index=None,
                    column=None,
                    rule="required_columns",
                    value=missing,
                    message=f"Missing required columns: {missing}",
                )
            )
            # If schema is wrong, stop here. Row validation makes no sense.
            return issues

        # ---- 2) Row-level validation ----
        # We keep checks readable and explicit: good for your report.
        for i, row in df.iterrows():
            # product_id: ASIN-like rule (10 alphanumeric chars)
            pid = (row.get("product_id") or "").strip()
            if not pid:
                issues.append(ValidationIssue(i, "product_id", "not_null", pid, "product_id is missing"))
            elif not ASIN_RE.match(pid):
                issues.append(ValidationIssue(i, "product_id", "asin_format", pid, "product_id not ASIN-like (10 alnum)"))

            # product_name: not null, min length
            name = (row.get("product_name") or "").strip()
            if len(name) < 3:
                issues.append(ValidationIssue(i, "product_name", "min_length_3", name, "product_name too short or missing"))

            # category: not null (often hierarchical, but we won't force '|', just require non-empty)
            cat = (row.get("category") or "").strip()
            if not cat:
                issues.append(ValidationIssue(i, "category", "not_null", cat, "category is missing"))

            # discounted_price & actual_price: parseable and > 0
            disc = _parse_money(row.get("discounted_price"))
            act = _parse_money(row.get("actual_price"))
            if disc is None or disc <= 0:
                issues.append(ValidationIssue(i, "discounted_price", "money_positive", row.get("discounted_price"), "Invalid discounted_price"))
            if act is None or act <= 0:
                issues.append(ValidationIssue(i, "actual_price", "money_positive", row.get("actual_price"), "Invalid actual_price"))
            if disc is not None and act is not None and disc > act:
                issues.append(ValidationIssue(i, "discounted_price", "discounted_le_actual", disc, "discounted_price > actual_price"))

            # discount_percentage: 0..100
            pct = _parse_percent(row.get("discount_percentage"))
            if pct is None or pct < 0 or pct > 100:
                issues.append(ValidationIssue(i, "discount_percentage", "percent_0_100", row.get("discount_percentage"), "Invalid discount_percentage"))

            # rating: 0..5 (some rows might be blank or non-numeric)
            try:
                r = float(str(row.get("rating")).strip()) if row.get("rating") not in (None, "") else None
            except ValueError:
                r = None
            if r is None or r < 0 or r > 5:
                issues.append(ValidationIssue(i, "rating", "rating_0_5", row.get("rating"), "Invalid rating"))

            # rating_count: int >= 0 (allow missing but flag it)
            rc_raw = row.get("rating_count")
            rc = _parse_int(rc_raw)
            if rc is None:
                issues.append(ValidationIssue(i, "rating_count", "int_or_missing", rc_raw, "rating_count missing or not parseable"))
            elif rc < 0:
                issues.append(ValidationIssue(i, "rating_count", "non_negative", rc, "rating_count < 0"))

            # product_link: valid URL
            plink = (row.get("product_link") or "").strip()
            if not _is_valid_url(plink):
                issues.append(ValidationIssue(i, "product_link", "valid_url", plink, "product_link not a valid URL"))

            # img_link: valid URL (optional: some datasets have blanks; we still validate if present)
            ilink = (row.get("img_link") or "").strip()
            if ilink and not _is_valid_url(ilink):
                issues.append(ValidationIssue(i, "img_link", "valid_url", ilink, "img_link not a valid URL"))

        return issues