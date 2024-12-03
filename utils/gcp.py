from google.cloud import storage
from google.auth import credentials
import pandas as pd
import io
import logging
import os


service_account = os.getenv("GCP_SERVICE_ACCOUNT")

def load_data_from_gcs(bucket_name: str, file_name: str, storage_client=None) -> pd.DataFrame:
    # Instantiate the Cloud Storage client
    if not storage_client:
        storage_client = storage.Client.from_service_account_json(service_account)
    try:
        # Get the bucket
        bucket = storage_client.bucket(bucket_name)

        # Get the blob (file)
        blob = bucket.blob(file_name)

        # Download the file into memory
        file_data = blob.download_as_text()

        # Use pandas to read the CSV data
        data = pd.read_csv(io.StringIO(file_data))
    except Exception as e:
        logging.error(f"Error loading data from GCS: {e}")
        data = None
    
    return data



def upload_data_to_gcs(bucket_name: str, file_name: str, data :str, storage_client=None) -> None:
    # Instantiate the Cloud Storage client
    if not storage_client:
        storage_client = storage.Client.from_service_account_json(service_account)
    
    try:
        # Get the bucket
        bucket = storage_client.bucket(bucket_name)

        # Create a blob (file)
        blob = bucket.blob(file_name)

        # Upload the CSV string to the bucket
        blob.upload_from_string(data.to_csv(index=False), content_type='text/csv')
    except Exception as e:
        logging.error(f"Error uploading data to GCS: {e}")
    
    return None


def append_row_to_csv(bucket_name: str, file_path :str, row_data, storage_client=None):
    # Initialize the storage client
    if not storage_client:
        storage_client = storage.Client.from_service_account_json(service_account)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    # Download the existing CSV file
    try:
        existing_data = load_data_from_gcs(bucket_name, file_path, storage_client)
    except Exception as e:
        # If the file does not exist, create a new DataFrame
        df = pd.DataFrame(columns=["datetime", "num_rows", "run_time", "num_processors"])

    # Convert the row data to a DataFrame
    new_row = pd.DataFrame([row_data.split(", ")], columns=["datetime", "num_rows", "run_time", "num_processors"])

    # Append the new row to the existing DataFrame
    df = pd.concat([existing_data, new_row], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    upload_data_to_gcs(bucket_name, file_path, df, storage_client)
    
    return None


def add_feedback_to_csv(bucket_name: str, file_path :str, row_data: list, storage_client=None):
    # Initialize the storage client
    if not storage_client:
        storage_client = storage.Client.from_service_account_json(service_account)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    # Download the existing CSV file
    try:
        existing_data = load_data_from_gcs(bucket_name, file_path, storage_client)
    except Exception as e:
        # If the file does not exist, create a new DataFrame
        existing_data = pd.DataFrame(columns=["datetime", "input", "probas", "feedback"])

    # Convert the row data to a DataFrame
    new_row = pd.DataFrame([row_data], columns=["datetime", "input", "probas", "feedback"])

    # Append the new row to the existing DataFrame
    df = pd.concat([existing_data, new_row], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    upload_data_to_gcs(bucket_name, file_path, df, storage_client)
    
    return

