# src/processor.py
from __future__ import annotations

import pandas as pd
from src.utils import clean_text, parse_money, parse_percent, parse_int


class Processor:
    """
    Cleans and enriches the dataset.
    - Removes duplicates
    - Adds extra columns (>= 3)
    """

    def __init__(self, dedup_subset: list[str] | None = None):
        self.dedup_subset = dedup_subset or ["product_id"]

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # 1) Trim common text columns
        for col in ["product_id", "product_name", "category"]:
            if col in out.columns:
                out[col] = out[col].apply(clean_text)

        # 2) Remove duplicates
        subset = [c for c in self.dedup_subset if c in out.columns]
        if subset:
            out = out.drop_duplicates(subset=subset, keep="first")

        # 3) Create numeric columns (keep original columns too)
        out["discounted_price_num"] = out["discounted_price"].apply(parse_money)
        out["actual_price_num"] = out["actual_price"].apply(parse_money)
        out["discount_pct_num"] = out["discount_percentage"].apply(parse_percent)
        out["rating_count_num"] = out["rating_count"].apply(parse_int)

        # 4) Derived columns (these are great for your assignment write-up)
        out["discount_amount"] = out["actual_price_num"] - out["discounted_price_num"]

        if "review_content" in out.columns:
            out["review_word_count"] = (
                out["review_content"]
                .fillna("")
                .astype(str)
                .apply(lambda x: len(x.split()))
            )
        else:
            out["review_word_count"] = 0

        if "category" in out.columns:
            out["category_level1"] = out["category"].apply(lambda x: x.split("|")[0].strip() if x else "")
        else:
            out["category_level1"] = ""

        # Optional: simple flag column (useful for analytics + validation)
        out["has_discount_flag"] = out["discount_amount"].fillna(0) > 0

        return out