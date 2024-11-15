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


# Load data from local storage
data_dir = os.path.join(project_root, 'datasets')
data_file = os.path.join(data_dir, 'raw/dataset_full.csv')
df = pd.read_csv(data_file)


# Create a train-test split
df_training = pd.concat([df[df.label == 0].sample(n=(80500)), df[df.label != 0]])
df_testing = df[~df.index.isin(df_training.index)]
df_training.reset_index(drop=True, inplace=True)
df_testing.reset_index(drop=True, inplace=True)

# Load the data to local storage
df_training.to_csv("datasets/raw/training.csv", index=False)
df_testing.to_csv("datasets/raw/testing.csv", index=False)

# load the data to GCS
bucket_name = os.getenv("GCP_BUCKET_NAME")
upload_data_to_gcs(bucket_name, "data/raw/training.csv", df_training, client)
upload_data_to_gcs(bucket_name, "data/raw/testing.csv", df_testing, client)