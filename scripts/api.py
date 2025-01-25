import os, sys
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

# Import libraries
from fastapi import FastAPI
import tensorflow as tf
from transformers import BertTokenizer
import pandas as pd
from typing import Dict
from functools import partial 

# Import utilities
from utils.utils import build_bert_model, build_bert_dataset, clean_data
from utils.custom_metrics import (
    WeightedCategoricalCrossEntropy, 
    PrecisionMulticlass, 
    RecallMulticlass,
    F1ScoreMulticlass,
    weights
)


# Load model
## Load BERT model
def load_bert_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer

def load_bert_model():
    loss = WeightedCategoricalCrossEntropy(weights)
    metrics = [PrecisionMulticlass(name='precision'),
               RecallMulticlass(name='recall'),
               F1ScoreMulticlass(name='f1')]
    bert_model = build_bert_model(loss, metrics)
    bert_model.load_weights(os.path.join(project_root, "models", "bert_dummy.h5"))
    return bert_model

tokenizer = load_bert_tokenizer()
bert_model = load_bert_model()


# Helper functions
def predict_bert(input, model):
    input = build_bert_dataset(input, tokenizer)
    probas = model.predict(input)
    prediction = tf.argmax(probas, axis=1).numpy()
    return int(prediction[0]), probas[0].tolist()
    



# FASTAPI
app = FastAPI(
    title="Text Classification API",
    description="An API for text classification using a BERT-based model.",
    version="1.0.0"
)


# API Endpoint
@app.post("/predictbert", summary="Predict text classification", tags=["Prediction"])
async def classify_text(input_data: Dict[str, str]):
    """
    Accepts a string of text as input and returns the classification result.
    
    **Input:** JSON with `text` field.
    **Output:** JSON with `prediction` field.
    """
    text = input_data.get("text", "")
    if not text:
        return {"error": "No text provided for classification."}
    
    
    input = pd.DataFrame({"text": [text]}) # Useful for prediction purpose
    input = clean_data(input)
    
    # Make a prediction
    prediction = predict_bert(input, bert_model)
    return {"prediction": prediction[0], "probas": prediction[1]}
