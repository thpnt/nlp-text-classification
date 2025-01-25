import sys, os, re, signal
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import TFBertModel
from textblob import TextBlob

# Custom utilities
from utils.artifacts import REGEX_REMOVE, REGEX_REPLACE, slang_dict


# NLTK resources
import nltk
from nltk.corpus import wordnet, words, webtext, gutenberg, brown
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('words', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('webtext', quiet=True)
nltk.download('gutenberg', quiet=True)
nltk.download('brown', quiet=True)
corpus = (set(words.words()) | set(wordnet.words()) | 
          set(webtext.words()) | set(gutenberg.words()) | 
          set(brown.words()))
combined_corpus = {word.lower() for word in corpus}





def clean_data(batch, slang_dict=slang_dict):
    """
    Cleans and preprocesses the data in the given batch.
    This function performs text cleaning, slang correction, and other preprocessing steps
    on the 'text' column of the input batch. The cleaned and corrected text is stored in 
    a new column 'corrected_text'.
    
    Parameters:
    batch (pd.DataFrame): The input data batch containing a 'text' column.
    slang_dict (dict): A dictionary for slang correction. Default is slang_dict.
    
    Returns:
    pd.DataFrame: The processed batch with an additional 'corrected_text' column.
    """
    def clean_text(text: str) -> str:
        # Apply REGEX_REMOVE and REGEX_REPLACE
        for pattern in REGEX_REMOVE:
            text = re.sub(pattern, "", text)
        for pattern, repl in REGEX_REPLACE.items():
            text = re.sub(pattern, repl, text)
        
        # Apply additionnal text cleaning steps
        text = re.sub(r'^RT @\w+: ', '', text)
        text = re.sub(r'http\S+', ' ', text)
        text = re.sub(r'\b\w*jpeg\w*\b|\b\w*jpg\w*\b', '', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'@\w+', '<PERSON>', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\b(\w+)\b\s+\1\b', '', text)
        text = text.strip().lower()
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'[\x80-\xFF]', '', text)
        return text

        
    def correct_text(text: str, slang_dict: dict) -> str:
        tokens = text.split()
        tokens = [slang_dict.get(word, word) for word in tokens] # Replace slang
        tokens = [word for word in tokens if len(word) < 15] # Useful for TextBlob
        text = ' '.join(tokens)
        corrected_text = str(TextBlob(text).correct())
        return corrected_text
    
    
    def combined_cleaning(text: str) -> list:
        text = clean_text(text)
        corrected_text = correct_text(text, slang_dict)
        return corrected_text
        
    # Process each text in the batch
    batch['corrected_text'] = batch['text'].apply(combined_cleaning)
    return batch
    
    
def build_gru_dataset(data: pd.DataFrame, batch_size:int = 512):
    """
    Helper function to build a tf.data.Dataset from the given data.
    """
    X = data["corrected_text"]
    dataset = tf.data.Dataset.from_tensor_slices((X))
    return dataset


def build_bert_model(loss: list, metrics: list, name:str = "bert_model"):
    """
    Helper function to build a BERT-based model for text classification.
    """
    bert_model = TFBertModel.from_pretrained('bert-base-uncased') # Load pre-trained model

    for layer in bert_model.layers:  # Freeze all layers
        layer.trainable = False


    input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="attention_mask")

    bert_outputs = bert_model(input_ids, attention_mask=attention_mask) # Retrieve BERT output
    pooled_output = bert_outputs.pooler_output                          # Use pooled output for classification

    # Add custom layers
    x = tf.keras.layers.Dense(128, activation='relu')(pooled_output)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output = tf.keras.layers.Dense(2, activation='softmax')(x)


    model = tf.keras.Model(inputs=[input_ids, attention_mask], 
                           outputs=output,
                           name=name)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
                  loss=loss,
                  metrics=metrics)

    print(model.summary())
    
    return model

def build_bert_dataset(data: pd.DataFrame, tokenizer, max_length=128, batch_size=512):
    """
    Helper function to build a tf.data.Dataset from the given data using the BERT tokenizer.
    """
    
    texts = data["text"].tolist()

    # Tokenize and encode the data
    encoded_data = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )

    dataset = tf.data.Dataset.from_tensor_slices((
        dict(encoded_data)
    ))

    return dataset.batch(batch_size)