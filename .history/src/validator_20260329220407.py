from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Callable
import pandas as pd

from src.utils import (
    clean_text,
    is_valid_asin,
    is_valid_url,
    parse_money,
    parse_percent,
    parse_int,
)


# ============================================================================
# EXPLICIT VALIDATION RULES FOR EACH COLUMN
# ============================================================================
# This configuration clearly documents what validation rules apply to each column
# making the data quality requirements visible and maintainable.
# ============================================================================

VALIDATION_RULES = {
    "product_id": {
        "description": "Amazon ASIN product identifier",
        "rules": [
            {"type": "not_empty", "message": "product_id cannot be empty"},
            {"type": "asin_format", "message": "product_id must be valid 10-character ASIN format"},
        ]
    },
    "product_name": {
        "description": "Product name/title",
        "rules": [
            {"type": "min_length", "value": 3, "message": "product_name must be at least 3 characters"},
            {"type": "max_length", "value": 500, "message": "product_name cannot exceed 500 characters"},
        ]
    },
    "category": {
        "description": "Product category",
        "rules": [
            {"type": "not_empty", "message": "category cannot be empty"},
            {"type": "max_length", "value": 200, "message": "category cannot exceed 200 characters"},
        ]
    },
    "discounted_price": {
        "description": "Current selling price",
        "rules": [
            {"type": "money_positive", "message": "discounted_price must be a positive amount"},
            {"type": "max_value", "value": 1000000, "message": "discounted_price seems unreasonably high"},
        ]
    },
    "actual_price": {
        "description": "Original/list price",
        "rules": [
            {"type": "money_positive", "message": "actual_price must be a positive amount"},
            {"type": "max_value", "value": 1000000, "message": "actual_price seems unreasonably high"},
        ]
    },
    "discount_percentage": {
        "description": "Discount percentage (0-100)",
        "rules": [
            {"type": "range", "min": 0, "max": 100, "message": "discount_percentage must be between 0 and 100"},
        ]
    },
    "rating": {
        "description": "Product rating (0-5 stars)",
        "rules": [
            {"type": "range", "min": 0, "max": 5, "message": "rating must be between 0 and 5"},
        ]
    },
    "rating_count": {
        "description": "Number of ratings",
        "rules": [
            {"type": "non_negative", "message": "rating_count must be >= 0"},
            {"type": "integer", "message": "rating_count must be an integer"},
        ]
    },
    "review_content": {
        "description": "Review text content (optional)",
        "rules": [
            {"type": "max_length", "value": 10000, "message": "review_content exceeds maximum length"},
        ]
    },
    "product_link": {
        "description": "URL to product page",
        "rules": [
            {"type": "not_empty", "message": "product_link cannot be empty"},
            {"type": "valid_url", "message": "product_link must be a valid HTTP(S) URL"},
        ]
    },
}


@dataclass
class ValidationIssue:
    row_index: int | None
    column: str
    rule: str
    message: str
    value: str = ""

    def to_dict(self) -> dict:
        return {
            "row_index": self.row_index,
            "column": self.column,
            "rule": self.rule,
            "message": self.message,
            "value": self.value,
        }


class Validator:
    """
    Validates the input DataFrame.
    Returns a list of ValidationIssue objects.
    If the list is empty => data is valid.
    """

    def __init__(self, required_columns: list[str]):
        self.required_columns = required_columns

    def validate(self, df: pd.DataFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # 1) Required columns
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            issues.append(
                ValidationIssue(
                    row_index=None,
                    column="__schema__",
                    rule="required_columns",
                    message=f"Missing required columns: {missing}",
                    value=",".join(missing),
                )
            )
            return issues

        # 2) Row checks
        for i, row in df.iterrows():
            self._validate_row(i, row, issues)

        return issues

    def _validate_row(self, i: int, row: pd.Series, issues: list[ValidationIssue]) -> None:
        # product_id
        pid = clean_text(row.get("product_id"))
        if not pid:
            issues.append(ValidationIssue(i, "product_id", "not_empty", "product_id is missing"))
        elif not is_valid_asin(pid):
            issues.append(ValidationIssue(i, "product_id", "asin_format", "product_id should be 10 letters/numbers", pid))

        # product_name
        name = clean_text(row.get("product_name"))
        if len(name) < 3:
            issues.append(ValidationIssue(i, "product_name", "min_length", "product_name is too short", name))

        # category
        cat = clean_text(row.get("category"))
        if not cat:
            issues.append(ValidationIssue(i, "category", "not_empty", "category is missing"))

        # prices
        disc = parse_money(row.get("discounted_price"))
        act = parse_money(row.get("actual_price"))

        if disc is None or disc <= 0:
            issues.append(ValidationIssue(i, "discounted_price", "money_positive", "discounted_price must be > 0", clean_text(row.get("discounted_price"))))

        if act is None or act <= 0:
            issues.append(ValidationIssue(i, "actual_price", "money_positive", "actual_price must be > 0", clean_text(row.get("actual_price"))))

        if disc is not None and act is not None and disc > act:
            issues.append(ValidationIssue(i, "discounted_price", "discounted_le_actual", "discounted_price cannot be higher than actual_price", str(disc)))

        # discount %
        pct = parse_percent(row.get("discount_percentage"))
        if pct is None or pct < 0 or pct > 100:
            issues.append(ValidationIssue(i, "discount_percentage", "range_0_100", "discount_percentage must be between 0 and 100", clean_text(row.get("discount_percentage"))))

        # rating
        r_txt = clean_text(row.get("rating"))
        try:
            r = float(r_txt) if r_txt else None
        except ValueError:
            r = None

        if r is None or r < 0 or r > 5:
            issues.append(ValidationIssue(i, "rating", "range_0_5", "rating must be between 0 and 5", r_txt))

        # rating_count (allow missing but still report it)
        rc_raw = row.get("rating_count")
        rc = parse_int(rc_raw)
        if rc is None:
            issues.append(ValidationIssue(i, "rating_count", "missing_or_invalid", "rating_count is missing or invalid", clean_text(rc_raw)))
        elif rc < 0:
            issues.append(ValidationIssue(i, "rating_count", "non_negative", "rating_count must be >= 0", str(rc)))

        # links
        plink = clean_text(row.get("product_link"))
        if not plink or not is_valid_url(plink):
            issues.append(ValidationIssue(i, "product_link", "valid_url", "product_link must be a valid http(s) URL", plink))

        ilink = clean_text(row.get("img_link"))
        if ilink and not is_valid_url(ilink):
            issues.append(ValidationIssue(i, "img_link", "valid_url", "img_link must be a valid http(s) URL", ilink))