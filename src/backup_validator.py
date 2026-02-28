# src/backup_validator.py
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class BackupValidationIssue:
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


class BackupValidator:
    """
    Back-up validator checks the output AFTER processing.
    Why? Because transformations can create new problems:
    - numeric conversion becomes NaN
    - derived values become negative
    - discount_amount doesn't make sense
    """

    def validate(self, df: pd.DataFrame) -> list[BackupValidationIssue]:
        issues: list[BackupValidationIssue] = []

        # These columns should exist if Processor ran correctly
        required = [
            "discounted_price_num",
            "actual_price_num",
            "discount_pct_num",
            "discount_amount",
        ]

        for col in required:
            if col not in df.columns:
                issues.append(
                    BackupValidationIssue(
                        row_index=None,
                        column=col,
                        rule="missing_column",
                        message=f"Column '{col}' is missing after processing.",
                    )
                )
                # If processor output is missing expected columns, stop early
                return issues

        # 1) Check for NaNs in numeric columns (a sign parsing failed)
        for col in ["discounted_price_num", "actual_price_num", "discount_pct_num"]:
            bad_rows = df.index[df[col].isna()].tolist()
            if bad_rows:
                issues.append(
                    BackupValidationIssue(
                        row_index=bad_rows[0],
                        column=col,
                        rule="not_nan",
                        message=f"{col} has NaN values after processing (parsing failed for some rows).",
                        value=str(len(bad_rows)),
                    )
                )

        # 2) Check business logic: discounted <= actual, discount_amount >= 0
        bad = df.index[df["discount_amount"] < 0].tolist()
        if bad:
            issues.append(
                BackupValidationIssue(
                    row_index=bad[0],
                    column="discount_amount",
                    rule="non_negative",
                    message="discount_amount is negative (discounted_price_num > actual_price_num).",
                    value=str(df.loc[bad[0], "discount_amount"]),
                )
            )

        # 3) discount percentage should be 0..100
        bad = df.index[(df["discount_pct_num"] < 0) | (df["discount_pct_num"] > 100)].tolist()
        if bad:
            issues.append(
                BackupValidationIssue(
                    row_index=bad[0],
                    column="discount_pct_num",
                    rule="range_0_100",
                    message="discount_pct_num is outside 0..100 after processing.",
                    value=str(df.loc[bad[0], "discount_pct_num"]),
                )
            )

        return issues