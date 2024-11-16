# Add the parent directory to sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import packages
import os, sys, time, logging, csv
import pandas as pd
from multiprocessing import cpu_count
from dotenv import load_dotenv
from utils.preprocessing import TokenizerMP, run_logger, error_logger
from utils.gcp import load_data_from_gcs, upload_data_to_gcs, append_row_to_csv 

# Load environment variables
load_dotenv()

# Google Cloud Authentication
project_root = os.path.dirname(os.path.dirname(__file__))
service_account = os.path.join(project_root, 'setup', os.getenv("GCP_SERVICE_ACCOUNT"))
from google.auth import credentials
from google.cloud import storage
client = storage.Client.from_service_account_json(service_account)


# Script to test the TokenizerMP class
if __name__ == "__main__": 
    # Load the data from GCS
    bucket_name = os.getenv("GCP_BUCKET_NAME")
    file_name = "data/raw/training.csv"
    df = load_data_from_gcs(bucket_name, file_name, client)
    df = df.sample(600)
    run_logger.info(f"Data loaded. Number of rows: {df.shape[0]}")
    
    # Clean and Tokenize the data and log the running time
    tokenizer = TokenizerMP(batch_size=32)
    start_time = time.time()
    cleaned_data = tokenizer.clean(df)
    run_time = time.time() - start_time
    run_logger.info(f"Processing complete. Time taken: {run_time:.2f} seconds.")
    
    
    # Save the cleaned data and logfile
    datetime = time.strftime("%Y%m%d-%H%M%S")
    # Saved the cleaned data to GCS
    upload_data_to_gcs(bucket_name, f"data/processed/{datetime}__data.csv", cleaned_data, client)
    run_logger.info(f"Cleaned data saved to GCS")
    
    
    # Append logs to GCS
    try:
        append_row_to_csv(bucket_name, "logs/log_clean_tokenize.csv", 
                      f"{datetime}, {cleaned_data.shape[0]}, {run_time}, {cpu_count()-2}", client)
        run_logger.info("Log added to log_clean_tokenize.csv")
    except Exception as e:
        error_logger.error(f"Error writing logs to CSV file: {e}")
    
    
    run_logger.info("Processing complete.")