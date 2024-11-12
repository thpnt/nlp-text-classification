from google.cloud import storage
import pandas as pd
import io

def load_data_from_gcs(bucket_name, file_name):
    # Instantiate the Cloud Storage client
    storage_client = storage.Client()
    
    # Get the bucket
    bucket = storage_client.bucket(bucket_name)
    
    # Get the blob (file)
    blob = bucket.blob(file_name)
    
    # Download the file into memory
    file_data = blob.download_as_text()
    
    # Use pandas to read the CSV data
    data = pd.read_csv(io.StringIO(file_data))
    
    return data



def write_data_to_gcs(bucket_name, file_name, data):
    # Instantiate the Cloud Storage client
    storage_client = storage.Client()
    
    # Get the bucket
    bucket = storage_client.bucket(bucket_name)
    
    # Create a blob (file)
    blob = bucket.blob(file_name)
    
    # Convert the DataFrame to a CSV string and upload it
    csv_data = data.to_csv(index=False)
    
    # Upload the CSV string to the bucket
    blob.upload_from_string(csv_data, content_type='text/csv')


# Authentication
from google.auth import credentials
from google.cloud import storage

client = storage.Client.from_service_account_json('path-to-your-service-account-file.json')
