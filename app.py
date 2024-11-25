import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
project_root = os.path.dirname(__file__)

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.input_pipeline import clean_data, feature_ready
from utils.custom_metrics import WeightedCategoricalCrossEntropy, PrecisionMulticlass, RecallMulticlass



# Import data to cache
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
    return tf.keras.models.load_model(os.path.join(project_root, 'models', 'baseline_model'),
                                      custom_objects={"WeightedCategoricalCrossEntropy": WeightedCategoricalCrossEntropy,
                                                      "PrecisionMulticlass": PrecisionMulticlass,
                                                      "RecallMulticlass": RecallMulticlass})


# Load model
embedding_matrix, vocab, token_to_index, model = load_embedding_matrix(), load_vocab(), load_token_to_index(), load_model()
max_length = 170


# Streamlit app header
st.markdown("# 🌟 Welcome to the Toxicity Detector!")

# Get user input text
user_input = st.text_area("Enter your text for toxicity analysis", placeholder="e.g., 'I dislike your attitude.'")


with st.expander("What is this about ?"):
    st.write("Project description.")
    
with st.expander("How does it work ?"):
    st.write("How the app works.")



# Predict
if st.button("Predict"):
    # Prepare data
    data = pd.DataFrame({"text": [user_input]})
    data = clean_data(data)
    data = feature_ready(data, embedding_matrix, vocab, token_to_index, max_length)
    
    # Predict
    with st.spinner("Analyzing..."):
        probas = model.predict(data)
        prediction = tf.argmax(probas, axis=1).numpy()
    
    
    # Prediction probabilities
    labels = ["Not Toxic", "Toxic", "Very Toxic"]
    fig, ax = plt.subplots()
    ax.bar(labels, probas[0])
    ax.set_title("Prediction Confidence")
    st.pyplot(fig)
    
    
    # Display prediction
    if prediction == 0:
        st.write("This is not toxic at all.")
    elif prediction == 1:
        st.write("This is toxic. 'Heshima si utumwa.'")
    else:
        st.write("This is very toxic and insulting. If you cannot respect, you cannot love.")

feedback = st.radio("Was this prediction accurate?", ["Yes", "No"])
if feedback == "Yes":
    st.write("Thank you for your feedback!")
else:
    st.write("I'll work on improving the model accuracy!")

st.markdown("---")   
with st.expander("Hear more from me on my github"):
    st.markdown("[Visit my github](https://www.github.com/thpnt)")
    


