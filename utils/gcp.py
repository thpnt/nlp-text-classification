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


def append_row_to_csv(bucket_name: str, file_name: str, data: str, storage_client=None) -> None:
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
        
        # Append the new data to the existing data
        new_data = pd.read_csv(io.StringIO(data))
        existing_data = pd.read_csv(io.StringIO(file_data))
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        
        # Upload the combined data back to the bucket
        blob.upload_from_string(combined_data.to_csv(index=False), content_type='text/csv')
    except Exception as e:
        logging.error(f"Error appending data to CSV file in GCS: {e}")
    
    return None


