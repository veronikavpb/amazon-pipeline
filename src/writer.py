from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from azure.storage.blob import BlobServiceClient


@dataclass
class Writer:
    """
    Writer = last step:
    - writes cleaned data to local output folder
    - uploads same file to Azure Blob Storage

    We write ONE output file per input file, which keeps things simple and traceable.
    """
    local_output_dir: str
    azure_connection_string: str
    azure_container_name: str
    azure_blob_prefix: str = "processed"

    def write_local_csv(self, df: pd.DataFrame, filename: str) -> Path:
        out_dir = Path(self.local_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / filename
        df.to_csv(out_path, index=False)
        return out_path

    def upload_to_azure(self, local_file_path: str | Path, blob_name: str) -> None:
        """
        Uploads a local file to Azure Blob Storage.

        blob_name example: processed/amazon_dataset_clean.csv
        """
        local_path = Path(local_file_path)

        if not local_path.exists():
            raise FileNotFoundError(f"File not found for upload: {local_path}")

        service = BlobServiceClient.from_connection_string(self.azure_connection_string)
        container = service.get_container_client(self.azure_container_name)

        # Create container if it doesn't exist (safe in pipelines)
        try:
            container.create_container()
        except Exception:
            # container probably already exists; we ignore
            pass

        blob_client = container.get_blob_client(blob_name)

        with open(local_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=True)

    def write_all(self, df: pd.DataFrame, base_filename: str) -> dict:
        """
        Convenience method used by the Airflow DAG.
        Returns paths used (helpful for logs).
        """
        # Local output
        local_path = self.write_local_csv(df, base_filename)

        # Azure blob path (prefix + filename)
        blob_name = f"{self.azure_blob_prefix}/{base_filename}"
        self.upload_to_azure(local_path, blob_name)

        return {
            "local_path": str(local_path),
            "blob_name": blob_name,
            "container": self.azure_container_name,
        }