from google.cloud import storage
from google.auth import credentials
import pandas as pd
import io
import logging
import os


service_account = os.getenv("GCP_SERVICE_ACCOUNT")

def load_data_from_gcs(bucket_name: str, file_name: str, storage_client=None) -> pd.DataFrame:
    """
    Helper function to load data from GCS.
    """
    if not storage_client:
        storage_client = storage.Client.from_service_account_json(service_account)
       
        
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        file_data = blob.download_as_text()
        data = pd.read_csv(io.StringIO(file_data))
        
    except Exception as e:
        logging.error(f"Error loading data from GCS: {e}")
        data = None
    
    return data





def upload_data_to_gcs(bucket_name: str, file_name: str, data :str, storage_client=None) -> None:
    """
    Helper function to upload data to GCS.
    """
    if not storage_client:
        storage_client = storage.Client.from_service_account_json(service_account)

    
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.upload_from_string(data.to_csv(index=False), content_type='text/csv')
        
    except Exception as e:
        logging.error(f"Error uploading data to GCS: {e}")
    
    return None






def append_row_to_csv(bucket_name: str, file_path :str, row_data, storage_client=None):
    """
    Helper function to append a row to a CSV file in GCS.
    """
    if not storage_client:
        storage_client = storage.Client.from_service_account_json(service_account)
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    # Fetch existing data
    try:
        existing_data = load_data_from_gcs(bucket_name, file_path, storage_client)
    except Exception as e:
        existing_data = pd.DataFrame(columns=["datetime", 
                                   "num_rows",
                                   "run_time", 
                                   "num_processors"]) # Create DF if file does not exist


    new_row = pd.DataFrame([row_data.split(", ")], 
                           columns=["datetime", "num_rows", "run_time", "num_processors"])


    # Append & save updated data
    data = pd.concat([existing_data, new_row], ignore_index=True)
    
    upload_data_to_gcs(bucket_name, file_path, data, storage_client)
    
    return None






def load_feedback_to_gcp(bucket_name: str, file_path :str, row_data: list, storage_client=None):
    """
    Helper function to load feedback data to GCS.
    """
    if not storage_client:
        storage_client = storage.Client.from_service_account_json(service_account)
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    # Fetch existing data
    try:
        existing_data = load_data_from_gcs(bucket_name, file_path, storage_client)
    
    except Exception as e:
        existing_data = pd.DataFrame(columns=["datetime", "input", "probas", "feedback"])

    new_row = pd.DataFrame([row_data], columns=["datetime", "input", "probas", "feedback"])

    # Append & save updated data
    df = pd.concat([existing_data, new_row], ignore_index=True)
    upload_data_to_gcs(bucket_name, file_path, df, storage_client)
    
    return

