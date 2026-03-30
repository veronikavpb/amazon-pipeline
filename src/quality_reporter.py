from __future__ import annotations

import pandas as pd
from pathlib import Path
from datetime import datetime


class QualityReporter:
    """
    Generates a simple data quality report for the processed dataset.
    Reports are saved as text files for easy viewing.
    """

    def generate_report(self, df: pd.DataFrame, output_path: str | Path) -> str:
        """
        Generate a data quality report and save it to a file.
        Returns the path to the saved report.
        """
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("DATA QUALITY REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Rows: {len(df):,}")
        report_lines.append(f"Total Columns: {len(df.columns)}")
        report_lines.append("")

        # ====================================================================
        # SECTION 1: Column Completeness
        # ====================================================================
        report_lines.append("COLUMN COMPLETENESS")
        report_lines.append("-"*80)
        report_lines.append(f"{'Column':<30} {'Non-Null':<12} {'Null':<12} {'Completeness':<15}")
        report_lines.append("-"*80)
        
        for col in df.columns:
            non_null = df[col].notna().sum()
            null = df[col].isna().sum()
            completeness = (non_null / len(df)) * 100
            report_lines.append(f"{col:<30} {non_null:<12,} {null:<12,} {completeness:>6.1f}%")
        
        report_lines.append("")

        # ====================================================================
        # SECTION 2: Numeric Column Statistics
        # ====================================================================
        report_lines.append("NUMERIC COLUMNS - KEY STATISTICS")
        report_lines.append("-"*80)
        
        numeric_cols = [
            "discounted_price_num", "actual_price_num", "discount_amount",
            "rating_num", "rating_count_num", "review_word_count", "price_quality_score"
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                report_lines.append(f"\n{col}:")
                report_lines.append(f"  Min:    {df[col].min():.2f}")
                report_lines.append(f"  Max:    {df[col].max():.2f}")
                report_lines.append(f"  Mean:   {df[col].mean():.2f}")
                report_lines.append(f"  Median: {df[col].median():.2f}")
        
        report_lines.append("")

        # ====================================================================
        # SECTION 3: Categorical Distributions
        # ====================================================================
        report_lines.append("CATEGORICAL DISTRIBUTIONS")
        report_lines.append("-"*80)
        
        # Top categories
        if "category_level1" in df.columns:
            report_lines.append("\nTop 10 Categories:")
            top_cats = df["category_level1"].value_counts().head(10)
            for cat, count in top_cats.items():
                pct = (count / len(df)) * 100
                report_lines.append(f"  {cat[:50]:<50} {count:>6,} ({pct:>5.1f}%)")
        
        # Discount flag
        if "has_discount_flag" in df.columns:
            report_lines.append("\nDiscount Distribution:")
            with_discount = df["has_discount_flag"].sum()
            without = len(df) - with_discount
            report_lines.append(f"  With Discount:    {with_discount:>6,} ({with_discount/len(df)*100:>5.1f}%)")
            report_lines.append(f"  Without Discount: {without:>6,} ({without/len(df)*100:>5.1f}%)")
        
        report_lines.append("")

        # ====================================================================
        # SECTION 4: Data Quality Score
        # ====================================================================
        report_lines.append("DATA QUALITY SCORE")
        report_lines.append("-"*80)
        
        # Calculate overall completeness
        total_cells = len(df) * len(df.columns)
        non_null_cells = df.notna().sum().sum()
        overall_completeness = (non_null_cells / total_cells) * 100
        
        # Critical field completeness (product_id, product_name, category)
        critical_fields = ["product_id", "product_name", "category"]
        critical_complete = all(
            df[col].notna().all() for col in critical_fields if col in df.columns
        )
        
        report_lines.append(f"Overall Completeness:     {overall_completeness:.2f}%")
        report_lines.append(f"Critical Fields Complete: {'✓ YES' if critical_complete else '✗ NO'}")
        
        # Quality score (simple weighted average)
        quality_score = overall_completeness * 0.7 + (100 if critical_complete else 0) * 0.3
        
        report_lines.append(f"\nQuality Score: {quality_score:.1f}/100")
        
        if quality_score >= 90:
            grade = "EXCELLENT"
        elif quality_score >= 80:
            grade = "GOOD"
        elif quality_score >= 70:
            grade = "ACCEPTABLE"
        else:
            grade = "NEEDS IMPROVEMENT"
        
        report_lines.append(f"Grade: {grade}")
        
        report_lines.append("")
        report_lines.append("="*80)
        report_lines.append("END OF REPORT")
        report_lines.append("="*80)

        # Write report to file
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        
        return str(report_path)
