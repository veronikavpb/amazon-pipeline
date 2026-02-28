# src/processor.py
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


def _parse_money_series(s: pd.Series) -> pd.Series:
    # Convert '₹1,099' → 1099.0
    cleaned = (
        s.astype(str)
         .str.replace("₹", "", regex=False)
         .str.replace(",", "", regex=False)
         .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_percent_series(s: pd.Series) -> pd.Series:
    cleaned = (
        s.astype(str)
         .str.replace("%", "", regex=False)
         .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_int_series(s: pd.Series) -> pd.Series:
    cleaned = (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.strip()
    )
    nums = pd.to_numeric(cleaned, errors="coerce")
    return nums.round().astype("Int64")  # nullable integer dtype


@dataclass
class Processor:
    """
    Processor = transformation + enrichment:
    - create derived columns (>= 3 required by assignment)
    - remove duplicates
    - standardize types (after validation)
    """
    dedup_subset: list[str] = None

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # ---- 1) Normalize text fields (small but valuable cleaning) ----
        for col in ["product_id", "product_name", "category"]:
            if col in out.columns:
                out[col] = out[col].astype(str).str.strip()

        # ---- 2) Remove duplicates ----
        # If no dedup keys provided, choose a reasonable default for this dataset:
        # product_id + product_link are usually stable identifiers.
        subset = self.dedup_subset or [c for c in ["product_id", "product_link"] if c in out.columns]
        if subset:
            out = out.drop_duplicates(subset=subset, keep="first")

        # ---- 3) Create derived numeric columns (this satisfies "3+ extra cols") ----
        # Numeric conversions kept as separate columns to preserve original raw text too
        out["discounted_price_num"] = _parse_money_series(out["discounted_price"])
        out["actual_price_num"] = _parse_money_series(out["actual_price"])
        out["discount_pct_num"] = _parse_percent_series(out["discount_percentage"])
        out["rating_count_num"] = _parse_int_series(out["rating_count"])

        # Derived: discount amount (should be >= 0 ideally)
        out["discount_amount"] = out["actual_price_num"] - out["discounted_price_num"]

        # Derived: review length / word count (nice for analytics + shows real enrichment)
        if "review_content" in out.columns:
            out["review_word_count"] = (
                out["review_content"]
                .astype(str)
                .str.split()
                .apply(lambda x: len(x) if isinstance(x, list) else 0)
            )
        else:
            out["review_word_count"] = 0

        # Derived: category level 1 (first chunk before '|')
        if "category" in out.columns:
            out["category_level1"] = out["category"].astype(str).str.split("|").str[0].str.strip()
        else:
            out["category_level1"] = None

        # Optional: basic “quality flags” that help later validation/reporting
        out["has_discount_flag"] = out["discount_amount"].fillna(0) > 0

        return out