import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def load_data(file):
    """
    Load and prepare the dataset from a CSV file.

    :param file: Uploaded CSV file containing research papers
    :return: DataFrame with combined 'Title' and 'Abstract' columns into a 'text' column for further processing.
    Returns None if required columns are missing.
    """
    # Load the dataset
    df = pd.read_csv(file)
    # Check for required columns
    if 'Title' not in df.columns or 'Abstract' not in df.columns:
        st.error("The uploaded file must have 'title' and 'abstract' columns.")
        return None
    df['text'] = df['Title'] + " " + df['Abstract']
    return df


def preprocess_text(text):
    """
    Clean and preprocess the text data for analysis.

    :param text: String containing the text to preprocess.
    :return: Cleaned and lowercase text with special characters and numbers removed.
    """
    if isinstance(text, str):
        # Remove special characters, numbers, and extra spaces
        text = re.sub(r'\W+', ' ', text)
        text = re.sub(r'\d+', '', text)
        return text.lower()
    else:
        return ""  # Return an empty string if text is not a valid string


@st.cache_data
def filter_papers(df, target_terms, threshold=0.2):
    """
    Filter papers by computing similarity between paper content and target terms.

    :param df: DataFrame containing research papers with a 'processed_text' column.
    :param target_terms: List of terms relevant to deep learning in virology/epidemiology.
    :param threshold: Similarity score threshold for filtering papers.
    :return: A filtered DataFrame containing only papers with a similarity score above the threshold.
    """
    # Preprocess text column
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Combine target terms for filtering
    target_string = " ".join(target_terms)

    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])
    target_vector = tfidf_vectorizer.transform([target_string])

    # Compute cosine similarity and filter based on threshold
    similarity_scores = cosine_similarity(tfidf_matrix, target_vector).flatten()
    df['similarity_score'] = similarity_scores
    filtered_df = df[df['similarity_score'] >= threshold]
    return filtered_df


def classify_method(text):
    """
    Classify papers based on method types: 'text mining', 'computer vision', 'both', or 'other'.

    :param text: The processed text of the paper
    :return: A string indicating the method category: 'text mining', 'computer vision', 'both', or 'other'.
    """
    # Keywords to identify 'text mining' and 'computer vision' methods
    text_mining_keywords = ["text mining", "natural language processing", "NLP", "sequence analysis"]
    computer_vision_keywords = ["image processing", "computer vision", "CNN", "image segmentation"]

    # Convert text to lowercase for consistent matching
    text = text.lower()

    # Determine if keywords for each category are present in the text
    has_text_mining = any(keyword in text for keyword in text_mining_keywords)
    has_computer_vision = any(keyword in text for keyword in computer_vision_keywords)

    # Classify based on presence of keywords
    if has_text_mining and has_computer_vision:
        return "both"
    elif has_text_mining:
        return "text mining"
    elif has_computer_vision:
        return "computer vision"
    else:
        return "other"


def extract_methods(text):
    """
    Extract specific deep learning methods mentioned in the paper.

    :param text: The processed text of the paper
    :return: A string listing the methods found in the paper or 'Not specified' if none are found.
    """
    # Define common deep learning method names for extraction
    method_keywords = ["CNN", "LSTM", "transformer", "RNN", "autoencoder", "GAN"]
    # Identify which method keywords appear in the text
    found_methods = [method for method in method_keywords if method.lower() in text.lower()]
    # Join and return the found methods, or indicate 'Not specified' if none are found
    return ", ".join(found_methods) if found_methods else "Not specified"
