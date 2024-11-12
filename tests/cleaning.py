# Add the parent directory to sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import packages
import os, sys, time, logging, csv
import pandas as pd
from multiprocessing import cpu_count
from dotenv import load_dotenv
from preprocessing.preprocessing import TokenizerMP, run_logger, error_logger
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
    # Load the data
    #project_root = os.path.dirname(os.path.dirname(__file__))
    #data_dir = os.path.join(project_root, 'datasets')
    #data_file = os.path.join(data_dir, 'raw/merged_dataset.csv')
    #df = pd.read_csv(data_file)
    
    # Load the data from GCS
    bucket_name = os.getenv("GCP_BUCKET_NAME")
    file_name = "testdata/merged_dataset.csv"
    df = load_data_from_gcs(bucket_name, file_name, client)
    
    
    # Reduce data size for testing --- TO BE REMOVED WHEN RUNNING ON FULL DATASET
    df_balanced = pd.concat([df[df.label == 0].sample(n=30000), df[df.label != 0]])
    df = df_balanced.sample(frac=1).sample(frac=0.005)
    run_logger.info(f"Data loaded. Number of rows: {df.shape[0]}")
    
    # Clean and Tokenize the data and log the running time
    tokenizer = TokenizerMP(batch_size=64)
    start_time = time.time()
    cleaned_data = tokenizer.clean(df)
    run_time = time.time() - start_time
    run_logger.info(f"Processing complete. Time taken: {run_time:.2f} seconds.")
    
    
    # Save the cleaned data and logfile
    datetime = time.strftime("%Y%m%d-%H%M%S")
    #cleaned_data.to_csv(os.path.join(data_dir, f'processed/{datetime}__training_set.csv'), index=False)
    #run_logger.info(f"Cleaned data saved to {data_dir}/processed/{datetime}__training_set.csv")
    # Saved the cleaned data to GCS
    upload_data_to_gcs(bucket_name, f"testprocessed/processed/{datetime}__training_set.csv", cleaned_data, client)
    run_logger.info(f"Cleaned data saved to GCS")
    
    
    # Add logs to CSV file
    #try:
    #    with open(os.path.join(project_root, 'logs', 'preprocessing', 'log_clean_tokenize.csv'), mode='a', newline='') as file:
    #        fieldnames = ["datetime", "num_rows", "runtime", "num_processors"]  
    #        writer = csv.DictWriter(file, fieldnames=fieldnames)
    #        writer.writerow({'datetime': datetime, 'num_rows' :int(cleaned_data.shape[0]), 
    #               'runtime' : run_time, 'num_processors' : int(cpu_count()-2)})
    #        run_logger.info("Log added to log_clean_tokenize.csv")
    #except Exception as e:
    #    error_logger.error(f"Error writing logs to CSV file: {e}")
    # Append logs to GCS
    append_row_to_csv(bucket_name, "testlogs/log_clean_tokenize.csv", 
                      f"{datetime}, {cleaned_data.shape[0]}, {run_time}, {cpu_count()-2}", client)
    run_logger.info("Log added to log_clean_tokenize.csv")
    run_logger.info("Processing complete.")