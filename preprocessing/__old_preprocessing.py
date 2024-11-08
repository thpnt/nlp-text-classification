import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import pandas as pd
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from setup.utils_setup import slang_dict
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords, words, wordnet
from tqdm import tqdm
tqdm.pandas()

# NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('words', quiet=True)
stop_words = stopwords.words('english')
combined_corpus = set(words.words()) | set(wordnet.words())
combined_corpus = {word.lower() for word in combined_corpus}

class Tokenizer:
    def __init__(self, batch_size: int=256):
        # Define stopwords as a class attribute for easy access
        self.stop_words = set(stop_words)
        self.batch_size = batch_size

    def clean(self, data: pd.DataFrame, text_series: str="text") -> pd.DataFrame:
        """
        This function processes a pandas Series of text data, applying a series of 
        regular expression substitutions to clean and standardize the text. The 
        cleaning steps include:
        1. Removing retweet patterns (e.g., "RT @user:").
        2. Removing URLs, image formats (jpg, jpeg), and line breaks.
        3. Replacing mentions with a placeholder ("<PERSON>") and removing punctuation.
        4. Removing numbers, stripping whitespaces, and removing repeating words.
        5. Removing special characters and emojis.
        6. Correcting spelling and slang terms.
        
        Args:
            data (pd.DataFrame): A pandas DataFrame containing text data.
            text_series (str): The name of the column containing the text data.
        
        Returns:
        pd.Series: A pandas Series with cleaned text.
        """
        def batches(data: pd.DataFrame, batch_size=self.batch_size) -> list:
            """
            Create batches of data for processing.
            """
            return [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

        # Comprehensive text cleaning function
        def clean_text(text):
            # Step 1: Remove retweet patterns (RT @user:)
            text = re.sub(r'^RT @\w+: ', '', text)
            # Step 2: Remove URLs, image formats (jpg, jpeg), and line breaks
            text = re.sub(r'http\S+', ' ', text)
            text = re.sub(r'\b\w*jpeg\w*\b|\b\w*jpg\w*\b', '', text)
            text = re.sub(r'\n', ' ', text)
            # Step 3: Replace mentions with a placeholder and remove punctuation
            text = re.sub(r'@\w+', '<PERSON>', text)
            text = re.sub(r'[^\w\s]', '', text)
            # Step 4: Remove numbers, strip whitespaces, and remove repeating words
            text = re.sub(r'\d+', '', text)
            text = re.sub(r'\b(\w+)\b\s+\1\b', '', text)
            text = text.strip().lower()
            # Step 5: Remove special characters and emojis
            text = re.sub(r'[^\x00-\x7F]+', ' ', text)
            text = re.sub(r'[\x80-\xFF]', '', text)

            return text

    

        # Function to correct spelling and slang terms
        def correct_text(text, stop_words: set, slang_dict: dict = slang_dict)-> str:
            """
            Replace slang terms and correct spelling in the entire sentence.
            """
            # Step 0: Tokenize the text
            tokens = text.split()
            # Step 1: Replace slang terms
            tokens = [slang_dict.get(word, word) for word in tokens]
            # Step 2: Remove stopwords from tokens
            tokens = [word for word in tokens if word not in stop_words]
            # Replace text with the corrected tokens
            text = ' '.join(tokens)
            # Step 3: Correct spelling using TextBlob on the entire sentence
            corrected_text = str(TextBlob(text).correct())
            return corrected_text
        
        data_copy = data.copy()
        
        
        # Process data in batches and replace old text with cleaned text
        for batch in batches(data_copy, self.batch_size):
            cleaned_batch = batch[text_series].apply(clean_text)
            corrected_batch = cleaned_batch.progress_apply(lambda x: correct_text(x, self.stop_words, slang_dict))
            data_copy.loc[batch.index, text_series] = corrected_batch  # Assign with .loc to avoid warning
        
        return data_copy
    
    def tokenize(self, data: pd.DataFrame, text_series:str = "text") -> pd.Series:
        """
        Tokenize and lemmatize a pandas Series of text data.
        This function tokenizes the text in the provided pandas Series using NLTK's
        word_tokenize function and then lemmatizes the tokens using WordNetLemmatizer.
        Args:
            data (pd.DataFrame): A pandas DataFrame containing text data.
            text_series (str): The name of the column containing the text data.
        Returns:
            pd.Series: A pandas Series where each element is a list of lemmatized tokens.
        """

        def lemma_text(tokens: list) -> list:
            """
            Lemmatize tokens using WordNet.
            """
            lemmatizer = WordNetLemmatizer()
            return [lemmatizer.lemmatize(token, 'v') for token in tokens]
        
        def replace_unknown_tokens(tokens: list) -> list:
            return [token if token in combined_corpus else '<UNK>' for token in tokens]
        

        
        data_copy = data.copy()
        data_copy[text_series] = data_copy[text_series].apply(word_tokenize, preserve_line=True).apply(lemma_text).apply(replace_unknown_tokens)
        return data_copy