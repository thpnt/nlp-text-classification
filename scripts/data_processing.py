# Add the parent directory to sys.path
import os, sys, time, logging, csv
import pandas as pd
from multiprocessing import cpu_count

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
project_root = os.path.dirname(os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(override=True) # Load the environment variables


# Custom utility functions
from utils.cleaning import TokenizerMP, run_logger, error_logger
from utils.gcp import load_data_from_gcs, upload_data_to_gcs, append_row_to_csv

# Google Cloud
from google.cloud import storage
service_account = os.path.join(project_root, os.getenv("GCP_SERVICE_ACCOUNT"))
client = storage.Client.from_service_account_json(service_account)



if __name__ == "__main__": 
    
    bucket_name = os.getenv("GCP_BUCKET_NAME")
    file_name = os.getenv("GCP_RAW_DATA_PATH")
    
    data = load_data_from_gcs(bucket_name, file_name, client)
    run_logger.info(f"Data loaded. Number of rows: {data.shape[0]}")
    
    # Data Cleaning
    tokenizer = TokenizerMP(batch_size=64)
    start_time = time.time() # Timer for log purposes
    cleaned_data = tokenizer.clean(data)
    run_time = time.time() - start_time
    run_logger.info(f"Processing complete. Time taken: {run_time:.2f} seconds.")
    
    
    # Save data to GCS
    datetime = time.strftime("%Y%m%d-%H%M%S")
    upload_data_to_gcs(bucket_name, f"data/processed/{datetime}__data.csv", cleaned_data, client)
    run_logger.info(f"Cleaned data saved to GCS")
    
    
    # Save logs to GCS
    try:
        file_name = "logs/log_clean_tokenize.csv"
        row = f"{datetime}, {cleaned_data.shape[0]}, {run_time}, {cpu_count()-1}"
        
        append_row_to_csv(bucket_name,
                          file_name, 
                          row, 
                          client)
        run_logger.info("Log added to log_clean_tokenize.csv")
    
    except Exception as e:
        error_logger.error(f"Error writing logs to CSV file: {e}")
    
    
    run_logger.info("Processing complete.")