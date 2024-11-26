text1 = """
Toxic comments, insults, and hate speech eare becoming increasingly prevalent online, affecting both individuals and communities. This project aims to leverage AI and Deep Learning techniques to help people identify and understand toxicity in online communication. The primary purpose of this website is to empower users to evaluate messages—whether received or intended to be sent—for toxicity and potential harm. By providing this tool, I hope to encourage thoughtful online communication and promote a more respectful digital environment. Think before you act is the core philosophy behind this initiative."""


text2 = """
This project integrates multiple Data Science and Deep Learning components to detect toxicity and hate speech in text input. Here's a technical breakdown:

1. **Dataset Integration**:
   - I combined two publicly available datasets to train the model: a **Toxic Comment Dataset** and a **Hate Speech Dataset**. These datasets encompass a wide variety of toxic, offensive, and neutral language to improve the model's ability to discern different levels of toxicity.

2. **Data Processing & Feature Engineering**:
   - Text data was pre-processed using Natural Language Processing (NLP) techniques:
     - **Text Cleaning**: Removed special characters, numbers, and unnecessary symbols to clean the data.
     - **Lowercasing**: Converted all text to lowercase to maintain uniformity.
     - **Tokenization**: Split sentences into individual words (tokens) for processing.
     - **Stop Word Removal**: Eliminated common but irrelevant words (like 'and', 'the') that don't add much value to the context.
     - **Stemming & Lemmatization**: Reduced words to their base or root forms to handle variations of the same word (e.g., "running" to "run").
   - **Feature Engineering**:
     - Created a custom **embedding matrix** for word representation using pre-trained embeddings, allowing the model to understand the semantic meaning of words.
     - Managed out-of-vocabulary words with special tokens, ensuring that even unseen words are handled gracefully.
     - Structured the data into sequences of fixed length, using padding for shorter inputs and truncation for longer ones, to maintain consistent input size for the model.

3. **Model Development**:
   - Developed and trained a **custom LSTM (Long Short-Term Memory) neural network** from scratch:
     - The model architecture leverages **LSTM layers**, which are specifically designed for handling sequential data like text, to capture long-term dependencies and context within sentences.
     - Implemented techniques like **dropout layers** to reduce overfitting and improve generalization.
     - Fine-tuned the model using custom metrics, such as a weighted categorical loss, to better handle the class imbalance in the dataset.
   - Training was conducted on a diverse dataset with toxic, neutral, and insulting samples, enabling the model to distinguish between varying levels of toxicity accurately.
   """