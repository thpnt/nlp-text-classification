import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

# Custom utility functions
from utils.input_pipeline import clean_data, feature_ready
from utils.text import text1, text2
from utils.custom_metrics import (
    WeightedCategoricalCrossEntropy, 
    PrecisionMulticlass, 
    RecallMulticlass
)

# Add project root to sys path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
project_root = os.path.dirname(__file__)

# Cache loaded data to avoid reloading on each interaction
@st.cache_data
def load_embedding_matrix():
    return np.load(os.path.join(project_root, 'models', 'embedding_matrix_300.npy'))

@st.cache_data
def load_vocab():
    return list(np.load(os.path.join(project_root, 'models', 'vocab.npy'), allow_pickle=True))

@st.cache_data
def load_token_to_index():
    return np.load(os.path.join(project_root, 'models', 'token_to_index.npy'), allow_pickle=True).item()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        os.path.join(project_root, 'models', 'baseline_model'),
        custom_objects={
            "WeightedCategoricalCrossEntropy": WeightedCategoricalCrossEntropy,
            "PrecisionMulticlass": PrecisionMulticlass,
            "RecallMulticlass": RecallMulticlass
        }
    )

# Initialize state variables if not present
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None
if 'probas' not in st.session_state:
    st.session_state['probas'] = None
if 'labels' not in st.session_state:
    st.session_state['labels'] = ["Not Toxic", "Toxic", "Very Toxic"]

# Load model and necessary data
embedding_matrix = load_embedding_matrix()
vocab = load_vocab()
token_to_index = load_token_to_index()
model = load_model()
max_length = 170
prediction = None

# Main app layout with tabs
tab1, tab2, tab3 = st.tabs(["Home", "About", "Details"])




# -------------- Home Tab --------------
tab1.markdown("# Website title!")

# User input text area
user_input = tab1.text_area(
    "Enter your text and get feedback", 
    placeholder="e.g., 'I dislike your attitude.'"
)

# Prediction logic
if tab1.button("Predict"):
    # Prepare input data
    data = pd.DataFrame({"text": [user_input]})
    data = clean_data(data)
    data = feature_ready(data, embedding_matrix, vocab, token_to_index, max_length)

    # Predict and store in session state
    with st.spinner("Analyzing..."):
        probas = model.predict(data)
        prediction = tf.argmax(probas, axis=1).numpy()
        st.session_state['prediction'] = prediction
        st.session_state['probas'] = probas[0]

# Display prediction if available
if st.session_state['prediction'] is not None:
    prediction = st.session_state['prediction']
    probas = st.session_state['probas']
    
    # Show prediction results
    if prediction == 0:
        tab1.write("This is not toxic at all.")
    elif prediction == 1:
        tab1.write("This is toxic. 'Heshima si utumwa.'")
    else:
        tab1.write("This is very toxic and insulting. If you cannot respect, you cannot love.")

    # Feedback section
    feedback = tab1.radio("Was this prediction accurate?", ["Yes", "No"])
    if feedback == "Yes":
        tab1.write("Thank you for your feedback!")
    else:
        tab1.write("I'll work on improving the model accuracy!")

        
# -------------- Details Tab --------------
if st.session_state['probas'] is not None:
    # Plot prediction probabilities
    labels = ["Not Toxic", "Toxic", "Very Toxic"]
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
