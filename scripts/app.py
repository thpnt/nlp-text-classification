import sys, os, time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from transformers import BertTokenizer
from functools import partial   # Used to load tf.model with weigths parameter
from dotenv import load_dotenv
load_dotenv(override=True)

# Add project root to sys path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
project_root = os.path.dirname(os.path.dirname(__file__))

# Google Cloud
from google.cloud import translate_v2 as translate
from google.cloud import storage
from utils.gcp import load_feedback_to_gcp

bucket_name = os.getenv("GCP_BUCKET_NAME")
logs_path = os.getenv("GCP_LOGS_PATH")
service_account = os.path.join(project_root, os.getenv("GCP_SERVICE_ACCOUNT"))

storage_client = storage.Client.from_service_account_json(service_account)
translate_client = translate.Client.from_service_account_json(service_account)

# Custom utility functions
from utils.text import text1, text2
from utils.utils import (clean_data, build_gru_dataset, build_bert_dataset, 
                         build_bert_model)
from utils.custom_metrics import (
    WeightedCategoricalCrossEntropy, 
    PrecisionMulticlass, 
    RecallMulticlass,
    F1ScoreMulticlass,
    weights
)



# -------------- Load resources into cache --------------

@st.cache_resource
def load_gru_model():
    return tf.keras.models.load_model(
    os.path.join(project_root, "models", "bi_gru"),
    custom_objects={'PrecisionMulticlass': PrecisionMulticlass,
                    'RecallMulticlass': RecallMulticlass,
                    'F1ScoreMulticlass': F1ScoreMulticlass,
                    'WeightedCategoricalCrossEntropy': partial(WeightedCategoricalCrossEntropy, 
                                                               weights=weights)}
)

@st.cache_resource
def load_bert_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer
    
@st.cache_resource
def load_bert_model():
    loss = WeightedCategoricalCrossEntropy(weights)
    metrics = [PrecisionMulticlass(name='precision'),
               RecallMulticlass(name='recall'),
               F1ScoreMulticlass(name='f1')]
    bert_model = build_bert_model(loss, metrics)
    bert_model.load_weights(os.path.join(project_root, "models", "bert", "bert_model_test"))
    return bert_model




# APP LAYOUT
tab1, tab2, tab3 = st.tabs(["Home", "About", "Details"])


# -------------- Session state --------------

if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None
if 'probas' not in st.session_state:
    st.session_state['probas'] = None
if 'labels' not in st.session_state:
    st.session_state['labels'] = ["Not Toxic", "Toxic"]
if "feedback" not in st.session_state:
    st.session_state["feedback"] = None
if "user_input" not in st.session_state:
    st.session_state["user_input"] = None

# Load model and necessary data
gru_model = load_gru_model()
tokenizer = load_bert_tokenizer()
bert_model = load_bert_model()
prediction = None





# -------------- Home Tab --------------
tab1.markdown("# Is this text toxic ?")

# User input text area
user_input = tab1.text_area(
    "Enter your text and get feedback", 
    placeholder="e.g., 'I dislike your attitude.'"
)
        

# Model selection
model = tab1.selectbox("Select the model to use", ["BERT", "GRU"])

# Prediction logic
if tab1.button("Get the result"):
    
    
    # Detect language and translate if needed
    if user_input:
        try:
            lang = translate_client.detect_language(user_input)
            if lang["language"] != "en":
                translation = translate_client.translate(user_input, target_language="en")
                user_input = translation["translatedText"]
                tab1.write(f"Translated from {lang['language']} to English: {translation['translatedText']}")
        except:
            tab1.write("Error translating text. Please enter English text.")
    
    
    st.session_state["user_input"] = user_input
    
    
    input = pd.DataFrame({"text": [user_input]}) # Useful for prediction purpose
    input = clean_data(input)
    
    
    # Model Prediction
    if model == "GRU":
        input = build_gru_dataset(input).batch(1)

        with st.spinner("Analyzing..."):
            probas = gru_model.predict(input)
            prediction = tf.argmax(probas, axis=1).numpy()
            st.session_state['prediction'] = prediction
            st.session_state['probas'] = probas[0]


    elif model == "BERT":
        input = build_bert_dataset(input, tokenizer)
        
        with st.spinner("Analyzing..."):
            probas = bert_model.predict(input)
            prediction = tf.argmax(probas, axis=1).numpy()
            st.session_state['prediction'] = prediction
            st.session_state['probas'] = probas[0]

# Display prediction if available
if st.session_state['prediction'] is not None:
    
    prediction = st.session_state['prediction']
    probas = st.session_state['probas']
    
    if prediction == 0:
        tab1.write("This text appears to be non-toxic. Keep up the positive communication!")
    
    elif prediction == 1:
        tab1.write("The message is classfied as toxic.")
        

    # Feedback section
    feedback = tab1.radio("Was this prediction accurate?", ["Yes", "No"])
    
    if tab1.button("Submit feedback"):
        
        if feedback == "Yes":
            st.session_state["feedback"] = 1
            tab1.write("Thank you for your feedback!")
            
        elif feedback == "No":
            st.session_state["feedback"] = 0
            tab1.write("I'll work on improving the model accuracy!")
            
        # Save feedback to GCS
        row = [pd.Timestamp.now(), 
               st.session_state["user_input"], 
               st.session_state["probas"], 
               st.session_state["feedback"]]
        
        load_feedback_to_gcp(bucket_name, logs_path, row, storage_client)


    
# -------------- Details Tab --------------
# Plot prediction confidence (probabilities | softmax output)
if st.session_state['probas'] is not None:

    labels = ["Not Toxic", "Toxic"]
    
    fig, ax = plt.subplots(figsize=(7, 3))
    
    sns.set_theme(style="whitegrid")
    sns.set_color_codes("pastel")
    sns.barplot(x=st.session_state['probas'], y=st.session_state['labels'], ax=ax, color="g")
    
    ax.axvline(0.5, color='k', linestyle='--', lw=1)
    ax.grid(False)
    ax.set(xlim=(0, 1), ylabel="", xlabel="Prediction confidence")
    
    sns.despine(left=True, bottom=True)
    
    tab3.pyplot(fig)


# -------------- About Tab --------------
tab2.markdown("---")
tab2.markdown("## What is this about?")
tab2.write(f"{text1}")

tab2.markdown("---")
tab2.markdown("## How does it work?")
tab2.markdown(f"{text2}")

tab2.markdown("---")
tab2.markdown("### Check out more projects on my GitHub")
tab2.markdown("[Visit my GitHub](https://www.github.com/thpnt)")
