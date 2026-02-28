# src/reader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass
class CSVReader:
    """
    Reader = single responsibility:
    - only loads a file into a DataFrame
    - does NOT validate or clean (that belongs to Validator/Processor)

    This separation makes your pipeline easier to test and easier to explain in your report.
    """
    encoding: str | None = "utf-8"
    sep: str = ","

    def read(self, filepath: str | Path) -> pd.DataFrame:
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        if path.suffix.lower() != ".csv":
            raise ValueError(f"Unsupported file type: {path.suffix} (expected .csv)")

        # dtype=str keeps everything as text first, which is safer for messy CSVs.
        df = pd.read_csv(
            path,
            sep=self.sep,
            encoding=self.encoding,
            dtype=str,
            keep_default_na=True,  # keep NaNs
            na_values=["", "NA", "N/A", "null", "None"],
        )

        # Normalize column names by stripping whitespace
        df.columns = [c.strip() for c in df.columns]

        return df