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
        "description": "Number of ratings (optional)",
        "rules": [
            {"type": "non_negative", "message": "rating_count must be >= 0 if present"},
        ]
    },
    "review_content": {
        "description": "Review text content (optional, no length limit)",
        "rules": []
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

    def get_validation_rules_summary(self) -> str:
        """
        Returns a human-readable summary of all validation rules.
        Useful for documentation and debugging.
        """
        summary = ["VALIDATION RULES SUMMARY", "=" * 80]
        for col, config in VALIDATION_RULES.items():
            summary.append(f"\n{col.upper()}: {config['description']}")
            for rule in config['rules']:
                summary.append(f"  - {rule['message']}")
        return "\n".join(summary)

    def _validate_row(self, i: int, row: pd.Series, issues: list[ValidationIssue]) -> None:
        """
        Validates a single row against the VALIDATION_RULES configuration.
        Each column is checked against its defined rules.
        """
        # --- product_id ---
        pid = clean_text(row.get("product_id"))
        if not pid:
            issues.append(ValidationIssue(i, "product_id", "not_empty", 
                                         "product_id cannot be empty"))
        elif not is_valid_asin(pid):
            issues.append(ValidationIssue(i, "product_id", "asin_format", 
                                         "product_id must be valid 10-character ASIN format", pid))

        # --- product_name ---
        name = clean_text(row.get("product_name"))
        if len(name) < 3:
            issues.append(ValidationIssue(i, "product_name", "min_length", 
                                         "product_name must be at least 3 characters", name))
        if len(name) > 500:
            issues.append(ValidationIssue(i, "product_name", "max_length", 
                                         "product_name cannot exceed 500 characters", name[:50] + "..."))

        # --- category ---
        cat = clean_text(row.get("category"))
        if not cat:
            issues.append(ValidationIssue(i, "category", "not_empty", 
                                         "category cannot be empty"))
        elif len(cat) > 200:
            issues.append(ValidationIssue(i, "category", "max_length", 
                                         "category cannot exceed 200 characters", cat[:50] + "..."))

        # --- discounted_price ---
        disc = parse_money(row.get("discounted_price"))
        if disc is None or disc <= 0:
            issues.append(ValidationIssue(i, "discounted_price", "money_positive", 
                                         "discounted_price must be a positive amount", 
                                         clean_text(row.get("discounted_price"))))
        elif disc > 1000000:
            issues.append(ValidationIssue(i, "discounted_price", "max_value", 
                                         "discounted_price seems unreasonably high", str(disc)))

        # --- actual_price ---
        act = parse_money(row.get("actual_price"))
        if act is None or act <= 0:
            issues.append(ValidationIssue(i, "actual_price", "money_positive", 
                                         "actual_price must be a positive amount", 
                                         clean_text(row.get("actual_price"))))
        elif act > 1000000:
            issues.append(ValidationIssue(i, "actual_price", "max_value", 
                                         "actual_price seems unreasonably high", str(act)))

        # Cross-field validation: discounted <= actual
        if disc is not None and act is not None and disc > act:
            issues.append(ValidationIssue(i, "discounted_price", "discounted_le_actual", 
                                         "discounted_price cannot be higher than actual_price", 
                                         str(disc)))

        # --- discount_percentage ---
        pct = parse_percent(row.get("discount_percentage"))
        if pct is None or pct < 0 or pct > 100:
            issues.append(ValidationIssue(i, "discount_percentage", "range", 
                                         "discount_percentage must be between 0 and 100", 
                                         clean_text(row.get("discount_percentage"))))

        # --- rating ---
        # Allow non-numeric or missing values - processor will coerce to null
        r_txt = clean_text(row.get("rating"))
        if r_txt:  # Only validate if present and non-empty
            try:
                r = float(r_txt)
                if r < 0 or r > 5:
                    issues.append(ValidationIssue(i, "rating", "range", 
                                                 "rating must be between 0 and 5", r_txt))
            except ValueError:
                # Invalid format - let processor handle it
                pass

        # --- rating_count ---
        # Optional field - only validate if present and parseable
        rc_raw = row.get("rating_count")
        if pd.notna(rc_raw) and clean_text(rc_raw):  # Only check if not empty
            rc = parse_int(rc_raw)
            if rc is not None and rc < 0:
                issues.append(ValidationIssue(i, "rating_count", "non_negative", 
                                             "rating_count must be >= 0", str(rc)))

        # --- review_content (optional field, no validation needed) ---
        # No length restrictions - accept any review content

        # --- product_link ---
        plink = clean_text(row.get("product_link"))
        if not plink:
            issues.append(ValidationIssue(i, "product_link", "not_empty", 
                                         "product_link cannot be empty"))
        elif not is_valid_url(plink):
            issues.append(ValidationIssue(i, "product_link", "valid_url", 
                                         "product_link must be a valid HTTP(S) URL", plink))