from __future__ import annotations

import os
import glob
import shutil
from datetime import datetime

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
import time
from dotenv import load_dotenv

from src.reader import CSVReader
from src.validator import Validator
from src.processor import Processor
from src.backup_validator import BackupValidator
from src.writer import Writer


# Airflow container paths (match your Docker volume mount)
INPUT_DIR = "/opt/airflow/data/input"
OUTPUT_DIR = "/opt/airflow/data/output"
ERROR_DIR = "/opt/airflow/data/error"
ARCHIVE_DIR = "/opt/airflow/data/archive"

# Load env variables (place .env where Airflow can read it, or configure env in docker-compose)
load_dotenv()

REQUIRED_COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating",
    "rating_count",
    "review_content",
    "product_link",
]


def pick_latest_csv(**context) -> None:
    """
    Picks the most recently modified CSV file from the input directory.
    """
    files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not files:
        print("No CSV files found (should have been caught by poll_for_csv)")
        raise AirflowSkipException("No CSV files to process.")

    files.sort(key=os.path.getmtime)
    latest = files[-1]
    print(f"Selected file: {latest}")
    print(f"File size: {os.path.getsize(latest)} bytes")
    context["ti"].xcom_push(key="input_file", value=latest)


def validate_and_process(**context) -> None:
    input_file = context["ti"].xcom_pull(key="input_file")
    if not input_file:
        raise ValueError("No input file path found in XCom.")

    print(f"Starting pipeline for file: {input_file}")
    
    os.makedirs(ERROR_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Reader
    reader = CSVReader()
    df = reader.read(input_file)
    print(f"Read {len(df)} rows from {input_file}")

    # 2) Validator
    validator = Validator(required_columns=REQUIRED_COLUMNS)
    issues = validator.validate(df)
    print(f"Validation found {len(issues)} issue(s)")

    if issues:
        # Write validation log
        log_name = os.path.basename(input_file).replace(".csv", "_validation_errors.csv")
        log_path = os.path.join(ERROR_DIR, log_name)

        pd.DataFrame([i.to_dict() for i in issues]).to_csv(log_path, index=False)

        # Move bad input file to error folder
        shutil.move(input_file, os.path.join(ERROR_DIR, os.path.basename(input_file)))

        raise ValueError(f"Validation failed. Moved file to error folder. Log: {log_path}")

    # 3) Processor
    processor = Processor(dedup_subset=["product_id"])
    df_clean = processor.process(df)
    print(f"Processing complete: {len(df_clean)} rows after cleaning (down from {len(df)} rows)")

    # 3.5) Backup Validator - validates the processed data
    backup_validator = BackupValidator()
    backup_issues = backup_validator.validate(df_clean)
    print(f"Backup validation found {len(backup_issues)} issue(s)")

    if backup_issues:
        # Write backup validation log
        log_name = os.path.basename(input_file).replace(".csv", "_backup_validation_errors.csv")
        log_path = os.path.join(ERROR_DIR, log_name)

        pd.DataFrame([i.to_dict() for i in backup_issues]).to_csv(log_path, index=False)

        # Move processed file to error folder (processing failed validation)
        shutil.move(input_file, os.path.join(ERROR_DIR, os.path.basename(input_file)))

        raise ValueError(f"Backup validation failed. Processing created invalid data. Log: {log_path}")

    # Push cleaned df to XCom? (Not recommended: too big)
    # Instead: write output now in the same task, or write a temp file path.
    # We'll write output here and pass the output path forward.
    context["ti"].xcom_push(key="processed_rows", value=int(df_clean.shape[0]))
    context["ti"].xcom_push(key="clean_df_json_path", value="")  # placeholder


    # 4) Writer (local + azure)
    writer = Writer(
        local_output_dir=OUTPUT_DIR,
        azure_connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING", ""),
        azure_container_name=os.getenv("AZURE_CONTAINER_NAME", ""),
        azure_blob_prefix=os.getenv("AZURE_BLOB_PREFIX", "processed"),
    )

    if not writer.azure_connection_string or not writer.azure_container_name:
        raise ValueError("Azure config missing. Set AZURE_STORAGE_CONNECTION_STRING and AZURE_CONTAINER_NAME.")

    out_name = os.path.basename(input_file).replace(".csv", "_clean.csv")
    writer_info = writer.write_all(df_clean, out_name)

    # 5) Archive input file after success
    archive_path = os.path.join(ARCHIVE_DIR, os.path.basename(input_file))
    shutil.move(input_file, archive_path)
    print(f"SUCCESS: File archived to {archive_path}")

    # Log info in task logs
    print("Writer output:", writer_info)


# --- DAG and Task Definitions ---
def poll_for_csv(**context):
    """
    Polls the input directory for any .csv file.
    If none found, skips downstream tasks (not an error - just nothing to process).
    """
    print(f"Checking for CSV files in: {INPUT_DIR}")
    
    # Check if directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"WARNING: Input directory does not exist: {INPUT_DIR}")
        raise AirflowSkipException("Input directory does not exist. Nothing to process.")
    
    # List all files in directory for debugging
    all_files = os.listdir(INPUT_DIR)
    print(f"All files in directory: {all_files}")
    
    poll_seconds = 10  # Reduced from 60 to avoid long waits
    interval = 2       # Check every 2 seconds
    waited = 0
    
    while waited < poll_seconds:
        files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
        if files:
            print(f"Found {len(files)} CSV file(s): {files}")
            return  # Proceed to next task
        time.sleep(interval)
        waited += interval
    
    print("No CSV files found in input directory. Skipping this run.")
    raise AirflowSkipException("No CSV files to process. Will check again on next schedule.")

with DAG(
    dag_id="amazon_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="*/1 * * * *",  # every minute
    catchup=False,
    default_args={"retries": 0},
) as dag:
    poll_csv = PythonOperator(
        task_id="poll_for_csv",
        python_callable=poll_for_csv,
    )

    choose_file = PythonOperator(
        task_id="pick_latest_csv",
        python_callable=pick_latest_csv,
    )

    run_pipeline = PythonOperator(
        task_id="validate_process_write",
        python_callable=validate_and_process,
    )

    poll_csv >> choose_file >> run_pipeline