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
from src.quality_reporter import QualityReporter


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

    print("\n" + "="*80)
    print("STARTING PIPELINE")
    print("="*80)
    print(f"File: {os.path.basename(input_file)}")
    print(f"Size: {os.path.getsize(input_file):,} bytes")
    print("="*80 + "\n")
    
    os.makedirs(ERROR_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Reader
    reader = CSVReader()
    df = reader.read(input_file)
    print(f"✓ Read {len(df)} rows from {input_file}")

    # 2) Validator
    validator = Validator(required_columns=REQUIRED_COLUMNS)
    issues = validator.validate(df)
    print(f"✓ Validation: {len(issues)} issue(s) found")

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
    print(f"✓ Processing: {len(df_clean)} rows after cleaning ({len(df) - len(df_clean)} duplicates removed)")

    # 3.5) Backup Validator - validates the processed data
    backup_validator = BackupValidator()
    backup_issues = backup_validator.validate(df_clean)
    print(f"✓ Backup validation: {len(backup_issues)} issue(s) found")

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
    print(f"✓ Writer: Saved locally and uploaded to Azure")

    # 4.5) Generate Data Quality Report
    quality_reporter = QualityReporter()
    report_name = os.path.basename(input_file).replace(".csv", "_quality_report.txt")
    report_path = quality_reporter.generate_report(
        df_clean,
        os.path.join(OUTPUT_DIR, report_name)
    )
    print(f"✓ Quality Report: Generated {report_path}")

    # 5) Archive input file after success
    archive_path = os.path.join(ARCHIVE_DIR, os.path.basename(input_file))
    shutil.move(input_file, archive_path)
    
    # ========================================================================
    # PROCESSING SUMMARY - Shows up in Airflow logs
    # ========================================================================
    duplicates_removed = len(df) - len(df_clean)
    null_ratings = df_clean["rating_num"].isna().sum()
    null_rating_counts = df_clean["rating_count_num"].isna().sum()
    products_with_discount = df_clean["has_discount_flag"].sum()
    avg_discount = df_clean["discount_amount"].mean()
    
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"Input File:          {os.path.basename(input_file)}")
    print(f"Input Rows:          {len(df):,}")
    print(f"Output Rows:         {len(df_clean):,}")
    print(f"Duplicates Removed:  {duplicates_removed}")
    print(f"\nData Quality:")
    print(f"  Null Ratings:         {null_ratings} ({null_ratings/len(df_clean)*100:.1f}%)") 
    print(f"  Null Rating Counts:   {null_rating_counts} ({null_rating_counts/len(df_clean)*100:.1f}%)")
    print(f"  Products w/ Discount: {products_with_discount} ({products_with_discount/len(df_clean)*100:.1f}%)")
    print(f"  Avg Discount Amount:  ₹{avg_discount:.2f}")
    print(f"\nOutput Locations:")
    print(f"  Data:    {writer_info['local_path']}")
    print(f"  Report:  {report_path}")
    print(f"  Azure:   {writer_info['container']}/{writer_info['blob_name']}")
    print(f"  Archive: {archive_path}")
    print("="*80)
    print("✓ Pipeline completed successfully!")
    print("="*80 + "\n")


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