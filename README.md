# 🧠 Toxic Tweet Classifier

A machine learning pipeline for detecting toxic content in text. This project includes a data preprocessing pipeline deployed on Google Cloud, models trained in Jupyter notebooks, and an interactive web app built with Streamlit and Docker.


## 🧪 Running Locally (Streamlit Web App)

To launch the Streamlit app locally, run the following command:
```bash
uvicorn api:app --reload --port 8000
streamlit run scripts/app.py
```
⚠️ Make sure the trained model files (e.g., bert.h5) are saved in the correct path under models/.
The app sends text inputs to a FastAPI backend (running separately) to classify content using a BERT-based model.

## ⚙️ Data Preprocessing Pipeline
The data cleaning and preprocessing pipeline is built with multiprocessing to efficiently handle large datasets. Key features:

- Slang normalization
- Text cleaning using regex and NLP
- Lemmatization and token replacement

This pipeline is deployed on Google Cloud Run, pulling raw data from Google Cloud Storage (GCS), cleaning it, and saving the processed data back to GCS.


## 🧠 Model Training
Three models are trained in Jupyter notebooks (not included here):
- GRU-based
- LSTM-based
- BERT-based: Fine-tuned bert-base-uncased using custom loss and metrics

The BERT model uses: `WeightedCategoricalCrossEntropy` to handle class imbalance

Custom metrics: Precision, Recall, and F1-Score for multiclass tasks

Weights are saved as .h5 and loaded in the backend API for inference.

All models were evaluated with custom metrics (Precision, Recall, F1 Score) and are saved in the `models/` directory for deployment.

## 🚀 Streamlit App (Frontend)
Accessible via app.py, the app:

Accepts user text and detects its language

Translates non-English input into English using Google Translate API

Sends text to the FastAPI server for BERT-based prediction

Displays prediction and confidence scores

Collects user feedback and stores it in GCS

<!-- Optional: Add demo image -->

## 📦 Installation
```bash
git clone https://github.com/thpnt/nlp-text-classification.git
cd nlp-text-classification
pip install -r requirements.txt
```

## 🌐 Deployment
✅ Backend (api.py) and preprocessing scripts can be deployed with Cloud Run

✅ Streamlit frontend can be deployed via Docker or Streamlit Sharing

✅ .env manages access to GCP service accounts and paths

## 📬 Contact
Feel free to reach out or explore more projects:
github.com/thpnt


# NO PUBLIC USE POSSIBLE FOR NOW
The models are not saved in the repository for now.
Do not try to deploy the docker app without the models trained and saved.
I am working on making the models trained publicly available.
