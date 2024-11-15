# Add the parent directory to sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary libraries
import sys, re, time, logging, csv
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet, words
from textblob import TextBlob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import gc
tqdm.pandas()

# Import custom functions & artifacts
from utils.cleaning_items import slang_dict, REGEX_REMOVE, REGEX_REPLACE

from utils.logging_setup import logging_setup
# Logging configuration
run_logger, error_logger = logging_setup()

# NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('words', quiet=True)
nltk.download('punkt_tab', quiet=True)
stop_words = stopwords.words('english')
combined_corpus = set(words.words()) | set(wordnet.words())
combined_corpus = {word.lower() for word in combined_corpus}

# Helper function for multiprocessing with error handling
def process_batch(batch, stop_words, slang_dict):
    """
    Processes a batch of text data by cleaning, correcting, tokenizing, lemmatizing, and replacing unknown tokens.
    Args:
        batch (pd.DataFrame): A DataFrame containing a column 'text' with the text data to be processed.
        stop_words (set): A set of stop words to be removed from the text.
        slang_dict (dict): A dictionary mapping slang words to their corrected forms.
    Returns:
        pd.DataFrame: The processed batch with cleaned, corrected, tokenized, lemmatized text, and unknown tokens replaced.
    Raises:
        Exception: If an error occurs during processing, logs the error and returns the unmodified batch.
    """
    try:
        def clean_text(text: str) -> str:
            """
            Cleans the input text by applying several preprocessing steps.
            Args:
                text (str): The input text to be cleaned.
            Returns:
                str: The cleaned text.
            """
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
            """
            Corrects the given text by replacing slang words, removing stop words, and performing spelling correction.

            Args:
                text (str): The input text to be corrected.
                stop_words (set): A set of stop words to be removed from the text.
                slang_dict (dict): A dictionary where keys are slang words and values are their corresponding replacements.

            Returns:
                str: The corrected text after slang replacement, stop word removal, and spelling correction.
            """
            tokens = text.split()
            tokens = [slang_dict.get(word, word) for word in tokens]
            tokens = [word for word in tokens if word not in stop_words]
            text = ' '.join(tokens)
            corrected_text = str(TextBlob(text).correct())
            return corrected_text
        
        
        def lemma_text(tokens: list) -> list:
            """
            Lemmatizes a list of tokens using the WordNet lemmatizer.
            Args:
                tokens (list): A list of tokens (words) to be lemmatized.
            Returns:
                list: A list of lemmatized tokens.
            """
            lemmatizer = WordNetLemmatizer()
            return [lemmatizer.lemmatize(token, 'v') for token in tokens]
        
        def replace_unknown_tokens(tokens: list) -> list:
            """
            Replace tokens that are not in the combined_corpus with the '<UNK>' token.
            Args:
                tokens (list): A list of tokens to be processed.
            Returns:
                list: A list of tokens where unknown tokens are replaced with '<UNK>'.
            """
            return [token if token in combined_corpus else '<UNK>' for token in tokens]
        
        
        def combined_cleaning(text: str) -> list:
            """
            Combines all the cleaning steps into a single function.
            Args:
                text (str): The input text to be cleaned.
            Returns:
                list: A list of cleaned tokens.
            """
            text = clean_text(text)
            text = correct_text(text, stop_words, slang_dict)
            tokens = word_tokenize(text, preserve_line=True)
            tokens = lemma_text(tokens)
            tokens = replace_unknown_tokens(tokens)
            return tokens
        
        
        # Process each text in the batch
        batch['tokens'] = batch['text'].apply(combined_cleaning)
        return batch

    except Exception as e:
        error_logger.error(f"Error processing batch with index {batch.index[0]}-{batch.index[-1]}: {e}")
        return batch  # Return the batch unmodified if there's an error

class TokenizerMP:
    """
    A class used to tokenize and clean text data in parallel using multiprocessing.
    Attributes
    ----------
    stop_words : set
        A set of stop words to be removed from the text.
    batch_size : int
        The size of the batches in which the data will be split for parallel processing.
    Methods
    -------
    clean(data: pd.DataFrame, text_series: str = "text") -> pd.Series
        Cleans the text data by removing stop words and applying other preprocessing steps in parallel.
    """
    def __init__(self, batch_size: int = 256):
        self.stop_words = set(stop_words)
        self.batch_size = batch_size

    def clean(self, data: pd.DataFrame, text_series: str = "text") -> pd.Series:
        # Split data into batches
        batches = [data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size)]
        
        # Prepare arguments for parallel processing
        args = [(batch.copy(), self.stop_words, slang_dict) for batch in batches]
        
        # Use multiprocessing to process batches in parallel
        num_processors = cpu_count() - 1
        processed_batches = []
        
        # Process batches in parallel
        for i in range(0, len(args), num_processors):
            chunk = args[i:i + num_processors]
            with Pool(processes=num_processors) as pool:
                processed_batches.extend(pool.starmap(process_batch, chunk))
                
            # Cleanup the chunk to free memory
            del chunk
            gc.collect()
            
        final_df = pd.concat(processed_batches, ignore_index=True)
        # Cleanup 
        del processed_batches
        gc.collect()

        # Combine processed batches back into a single DataFrame
        return final_df



    