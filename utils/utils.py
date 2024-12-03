import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
from utils.artifacts import REGEX_REMOVE, REGEX_REPLACE
import numpy as np
from textblob import TextBlob
import signal
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet, words, webtext, gutenberg, brown
from utils.artifacts import slang_dict
from transformers import TFBertModel

import tensorflow as tf
#from tensorflow.keras.utils import pad_sequences

# NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('words', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('webtext', quiet=True)
nltk.download('gutenberg', quiet=True)
nltk.download('brown', quiet=True)
stop_words = stopwords.words('english')
combined_corpus = set(words.words()) | set(wordnet.words()) | set(webtext.words()) | set(gutenberg.words()) | set(brown.words())
combined_corpus = {word.lower() for word in combined_corpus}


# Project root
project_root = os.path.dirname(os.path.dirname(__file__))



def clean_data(batch, stop_words = stop_words, slang_dict=slang_dict):
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

        
        def correct_text(text: str, stop_words, slang_dict: dict) -> str:
            tokens = text.split()
            tokens = [slang_dict.get(word, word) for word in tokens]
            tokens = [word for word in tokens if word not in stop_words]
            #tokens = [word for word in tokens if len(word) < 15]
            text = ' '.join(tokens)
            corrected_text = str(TextBlob(text).correct())
            return corrected_text
        
        
        def replace_unknown_tokens(tokens: list) -> list:
            return [token if token in combined_corpus else '<UNK>' for token in tokens]
        
        
        def combined_cleaning(text: str) -> list:
            text = clean_text(text)
            corrected_text = correct_text(text, stop_words, slang_dict)
            return corrected_text
        
        # Process each text in the batch
        batch['corrected_text'] = batch['text'].apply(combined_cleaning)
        return batch
    
    
def build_gru_dataset(data: pd.DataFrame, batch_size:int = 512):
    # Prepare dataset
    X = data["corrected_text"]

    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X))

    return dataset


def build_bert_model(loss: list, metrics: list, name:str = "bert_model"):
    # Load the pre-trained BERT model
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')

    # Freeze the BERT model layers
    for layer in bert_model.layers:  # Freeze all layers
        layer.trainable = False

    # Define the input layers
    input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="attention_mask")

    # Get the output from the BERT model
    bert_outputs = bert_model(input_ids, attention_mask=attention_mask)

    # Use the pooled output for classification
    pooled_output = bert_outputs.pooler_output

    # Add custom layers
    x = tf.keras.layers.Dense(128, activation='relu')(pooled_output)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(2, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output, name=name)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
                  loss=loss,
                  metrics=metrics)

    # Summary of the model
    model.summary()
    
    return model

def build_bert_dataset(data: pd.DataFrame, tokenizer, max_length=128, batch_size=512):
    # Prepare dataset
    texts = data["text"].tolist()

    # Tokenize and encode the data
    encoded_data = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )

    # Create tf.data.Dataset with tokenized inputs
    dataset = tf.data.Dataset.from_tensor_slices((
        dict(encoded_data)
    ))

    # Batch the dataset
    return dataset.batch(batch_size)
    
    
# Import data
#embedding_matrix = np.load(os.path.join(project_root, 'models', 'embedding_matrix_300.npy'))
#vocab = list(np.load(os.path.join(project_root, 'models', 'vocab.npy'), allow_pickle=True))
#vocab.append("<PAD>")
#vocab_size = len(vocab)
#token_to_index = {token: idx for idx, token in enumerate(vocab)}
#max_length = 30
#    
#def feature_ready(data: pd.DataFrame, embedding_matrix: np.array, vocab: list, token_to_index: dict, max_length: int):
#    # Convert text tokens to index
#    #data['tokens_index'] = data['tokens'].apply(lambda x: [token_to_index.get(token, token_to_index["<UNK>"]) for token in x])
#    #pad = pad_sequences(data.tokens_index, maxlen=max_length, padding='post', truncating='post')
#    #data['padded_tokens'] = [list(row) for row in pad]
#    # Convert to tensor for inference
#    data_tensor = tf.ragged.constant(data["padded_tokens"], dtype=tf.int32)
#    data_tensor = data_tensor.to_tensor(default_value=0) 
#    return data_tensor