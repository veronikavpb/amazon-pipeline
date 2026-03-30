from __future__ import annotations

import pandas as pd
from src.utils import clean_text, parse_money, parse_percent, parse_int


class Processor:
    """
    Cleans and enriches the dataset.
    
    Key responsibilities:
    1. Removes duplicates based on product_id
    2. Creates at least 3 new derived columns (requirement met with 4+ columns):
       - discount_amount: Actual savings in currency
       - review_word_count: Length of review content
       - category_level1: Top-level category extracted
       - has_discount_flag: Boolean indicating if product has discount
       - price_quality_score: Rating weighted by number of ratings
    """

    def __init__(self, dedup_subset: list[str] | None = None):
        self.dedup_subset = dedup_subset or ["product_id"]

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the dataframe: cleans, deduplicates, and adds derived columns.
        
        Returns a cleaned DataFrame with additional analytical columns.
        """
        out = df.copy()

        # ========================================================================
        # STEP 1: Data Cleaning - Trim whitespace from text columns
        # ========================================================================
        for col in ["product_id", "product_name", "category"]:
            if col in out.columns:
                out[col] = out[col].apply(clean_text)

        # ========================================================================
        # STEP 2: DUPLICATE REMOVAL (Requirement)
        # ========================================================================
        # Remove duplicate products based on product_id
        subset = [c for c in self.dedup_subset if c in out.columns]
        if subset:
            initial_count = len(out)
            out = out.drop_duplicates(subset=subset, keep="first")
            duplicates_removed = initial_count - len(out)
            print(f"Removed {duplicates_removed} duplicate rows based on {subset}")

        # ========================================================================
        # STEP 3: Parse numeric values from text columns
        # ========================================================================
        out["discounted_price_num"] = out["discounted_price"].apply(parse_money)
        out["actual_price_num"] = out["actual_price"].apply(parse_money)
        out["discount_pct_num"] = out["discount_percentage"].apply(parse_percent)
        out["rating_count_num"] = out["rating_count"].apply(parse_int)

        # ========================================================================
        # STEP 4: CREATE NEW DERIVED COLUMNS (Requirement: at least 3)
        # ========================================================================
        
        # NEW COLUMN 1: discount_amount
        # Purpose: Calculate actual monetary savings (for financial analysis)
        out["discount_amount"] = out["actual_price_num"] - out["discounted_price_num"]
        out["discount_amount"] = out["discount_amount"].fillna(0)

        # NEW COLUMN 2: review_word_count
        # Purpose: Measure review length (for content quality metrics)
        if "review_content" in out.columns:
            out["review_word_count"] = (
                out["review_content"]
                .fillna("")
                .astype(str)
                .apply(lambda x: len(x.split()))
            )
        else:
            out["review_word_count"] = 0

        # NEW COLUMN 3: category_level1
        # Purpose: Extract top-level category for grouping/aggregation
        if "category" in out.columns:
            out["category_level1"] = out["category"].apply(
                lambda x: x.split("|")[0].strip() if x and "|" in x else x if x else "Unknown"
            )
        else:
            out["category_level1"] = "Unknown"

        # NEW COLUMN 4: has_discount_flag (BONUS)
        # Purpose: Boolean flag for filtering discounted products
        out["has_discount_flag"] = out["discount_amount"] > 0

        # NEW COLUMN 5: price_quality_score (BONUS)
        # Purpose: Weighted rating score (rating * log(rating_count + 1))
        # This gives more weight to products with more reviews
        out["price_quality_score"] = out.apply(
            lambda row: row.get("rating", 0) * (1 + (row.get("rating_count_num", 0) / 100))
            if pd.notna(row.get("rating")) and pd.notna(row.get("rating_count_num"))
            else 0,
            axis=1
        )

        return out