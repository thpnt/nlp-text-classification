# Imports
import sys, re, time, logging, csv, os, ast, gc, signal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from textblob import TextBlob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
tqdm.pandas()

# Import custom functions & artifacts
from utils.artifacts import slang_dict, REGEX_REMOVE, REGEX_REPLACE
from utils.logging import logging_setup

# NLTK resources
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, words, webtext, gutenberg, brown
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

# Logging configuration
run_logger, error_logger = logging_setup()

# Helper function for multiprocessing with error handling
def process_batch(batch, slang_dict):
    """
    Helper function, called by TokenizerMP.clean, to process a batch of text data in parallel.
    
    Processes a batch of text data by cleaning, correcting, tokenizing, lemmatizing, and replacing unknown tokens.
    Args:
        batch (pd.DataFrame): A DataFrame containing a column 'text' with the text data to be processed.
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

        def handler(signum, frame):
            raise TimeoutError("Process took too long")
        
        # Set the signal handler for the alarm
        signal.signal(signal.SIGALRM, handler)
        
        def correct_text(text: str, slang_dict: dict) -> str:
            """
            Corrects the given text by replacing slang words, removing stop words, and performing spelling correction.
            """
            tokens = text.split()
            tokens = [slang_dict.get(word, word) for word in tokens]
            tokens = [word for word in tokens if len(word) < 15]
            text = ' '.join(tokens)
            
            try:
                signal.alarm(3) # Set 3 seconds alarm
                corrected_text = str(TextBlob(text).correct())
                signal.alarm(0)
            
            except TimeoutError:
                corrected_text = text
            
            return corrected_text
        
        
        def lemma_text(tokens: list) -> list:
            """
            Lemmatizes a list of tokens using the WordNet lemmatizer.
            """
            lemmatizer = WordNetLemmatizer()
            
            ls = [lemmatizer.lemmatize(token, 'v') for token in tokens]
            ls = [lemmatizer.lemmatize(token, 'n') for token in ls]
            ls = [lemmatizer.lemmatize(token, 'a') for token in ls]
            
            return ls
        
        def replace_unknown_tokens(tokens: list) -> list:
            """
            Replace tokens that are not in the combined_corpus with the '<UNK>' token.
            """
            return [token if token in combined_corpus else '<UNK>' for token in tokens]
        
        
        def combined_cleaning(text: str) -> list:
            """
            Combines all the cleaning steps into a single function.
            """
            text = clean_text(text)
            corrected_text = correct_text(text, slang_dict)
            
            return corrected_text
        
        def tokenize(text: str) -> list:
            """
            Tokenizes the input text using the NLTK word_tokenize function.
            """
            tokens = word_tokenize(text, preserve_line=True)
            tokens = lemma_text(tokens)
            tokens = replace_unknown_tokens(tokens)
            return tokens
        
        # Process each text in the batch
        batch['corrected_text'] = batch['text'].apply(combined_cleaning)
        batch['tokens'] = batch['corrected_text'].apply(tokenize)
        return batch

    except Exception as e: # Catch and log any errors
        error_logger.error(f"Error processing batch with index {batch.index[0]}-{batch.index[-1]}: {e}")
        return batch  # Return the batch unmodified if there's an error








class TokenizerMP:
    """
    A class used to tokenize and clean text data in parallel using multiprocessing.
    Attributes
    ----------
    batch_size : int
        The size of the batches in which the data will be split for parallel processing.
    Methods
    -------
    clean(data: pd.DataFrame, text_series: str = "text") -> pd.Series
        Cleans the text data by removing stop words and applying other preprocessing steps in parallel.
    """
    
    def __init__(self, batch_size: int = 256, num_processors: int = None):
        self.batch_size = batch_size
        self.num_processors = num_processors if num_processors is not None else cpu_count()



    def clean(self, data: pd.DataFrame, text_series: str = "text") -> pd.Series:
        
        batches = [data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size)]
        
        # Prepare arguments for parallel processing
        args = [(batch.copy(), slang_dict) for batch in batches]
        
        # Use multiprocessing to process batches in parallel
        processed_batches = []
        total_rows = data.shape[0]
        processed_rows = 0
        
        for i in range(0, len(args), self.num_processors):
            chunk = args[i:i + self.num_processors]
            
            with Pool(processes=self.num_processors) as pool:
                processed_batches.extend(pool.starmap(process_batch, chunk))
                processed_rows += self.batch_size * self.num_processors
                
                # Log progress
                run_logger.info(f"Processed {processed_rows}/{total_rows} rows")
                
            del chunk # Clear memory
            gc.collect()
            
        data = pd.concat(processed_batches, ignore_index=True)

        del processed_batches # Clear memory
        gc.collect()

        return data



    