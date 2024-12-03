text1 = """
Toxic comments, insults, and hate speech eare becoming increasingly prevalent online, affecting both individuals and communities. This project aims to leverage AI and Deep Learning techniques to help people identify and understand toxicity in online communication. The primary purpose of this website is to empower users to evaluate messages—whether received or intended to be sent—for toxicity and potential harm. By providing this tool, I hope to encourage thoughtful online communication and promote a more respectful digital environment. Think before you act is the core philosophy behind this initiative."""


text2 = """
This project integrates multiple Deep Learning components to detect toxicity and hate speech in text input. Here's a technical breakdown:

1. **Dataset Integration**:
  - I combined two publicly available datasets to train the model: a **Toxic Comment Dataset** and a **Hate Speech Dataset**. These datasets encompass a wide variety of toxic, offensive, and neutral language to improve the model's ability to discern different levels of toxicity.
  - The training data is from web comments and messages. Therefore, the model is well-equipped to handle real-world online communication scenarios. **However, it's essential to note that the model's performance may vary based on the context and domain of the text input.**
  - At first training was conducted on a diverse dataset with toxic, neutral, and insulting samples, enabling the model to distinguish between varying levels of toxicity accurately.
     However, these small models were not able to capture all the nuances of toxiciy. Therefore, I decided to merge all toxic labels into one, resulting in a binary classification problem.

2. **Data Processing & Feature Engineering**:

**Data Processing**:
  - **Text Cleaning**: Removed special characters, numbers, and unnecessary symbols to clean the data.
  - **Lowercasing**: Converted all text to lowercase to maintain uniformity.
  - **Tokenization**: Split sentences into individual words (tokens) for processing.
     
**Feature Engineering**:
  - Created a custom **embedding matrix** for word representation using pre-trained embeddings, allowing the model to understand the semantic meaning of words.
  - Managed out-of-vocabulary words with special tokens, ensuring that even unseen words are handled gracefully.
  - Structured the data into sequences of fixed length, using padding for shorter inputs and truncation for longer ones, to maintain consistent input size for the model.

**Additionnal Steps**:

There are some additional steps that I did not include, as they were resulting in lower accuracy. 
  - **Stop Word Removal**: Removing stopwords was preventing the model from capturing the context and meaning of sentences effectively. Especially in the case of ambiguous message using negative form to express positive sentiment.
  - **Lemmatization**: Reducing words to their base form was causing a loss of information, especially in the case of slang or informal language.

3. **Model Development**:

I developped multiple models to compare performances : a **GRU-based model** and a **BERT-based model**.
  
   
**GRU Model**:

I developed and trained from scratch a **custom Bi-Directional GRU-based neural network**:
  - The model architecture leverages **GRU layers**, which are specifically designed for handling sequential data like text, to capture long-term dependencies and context within sentences.
  - I implemented techniques like **dropout layers** to reduce overfitting and improve generalization.
  - I fine-tuned the model using custom metrics, such as a weighted categorical loss, to better handle the class imbalance in the dataset.
  - Training was conducted on a diverse dataset with toxic, neutral, and insulting samples, enabling the model to distinguish between varying levels of toxicity accurately.
   
**BERT Model**:

I fine-tuned a pre-trained **BERT (Bidirectional Encoder Representations from Transformers)** model for text classification:
    - BERT is a state-of-the-art transformer-based model that captures the context and semantics of words in a sentence, enabling it to understand the meaning of text more effectively.
   
   """