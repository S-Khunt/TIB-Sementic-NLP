# Research Paper Filtering and Classification App

This application filters and classifies research papers based on their relevance to **deep learning applications in virology and epidemiology**. It provides a user-friendly interface for filtering relevant papers using a similarity-based approach, classifying them by research method, and extracting deep learning methods mentioned in each paper.

## Table of Contents
- [Overview](#overview)
- [Components](#components)
- [Installation](#installation)
- [How to Use](#how-to-use)
- [NLP Technique](#nlp-technique)
- [Approach Justification](#approach-justification)

---

## Overview

The app is designed to streamline the initial stages of academic literature review by using Natural Language Processing (NLP) techniques to:
1. **Filter** papers based on relevance to deep learning in virology/epidemiology.
2. **Classify** relevant papers by research method: "text mining," "computer vision," "both," or "other."
3. **Extract** specific deep learning methods mentioned in each paper.

This approach helps reduce the time and effort required for manual sorting of academic articles.

## Components

The app consists of the following components:

1. **Data Loader**: Loads a user-uploaded CSV file with required fields ("Title" and "Abstract").
2. **Text Preprocessing**: Cleans the text data to prepare it for filtering.
3. **Semantic Filtering**: Uses TF-IDF vectorization and cosine similarity to filter relevant papers.
4. **Classification by Method**: Classifies relevant papers by research methods based on predefined keywords.
5. **Method Extraction**: Extracts mentions of specific deep learning methods from the filtered papers.

These functions are organized into a `functions.py` file and are called from within the Streamlit app, providing an interactive interface.

## Installation

1. Clone the repository.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
3. Run as the Streamlit app:
    ```bash
   streamlit run app.py

## How to Use
1. **Upload CSV File**: The CSV should contain at least "Title" and "Abstract" columns for each paper.
2. **Select Target Terms**: Choose relevant terms from a predefined list or add custom terms. These terms represent deep learning and virology/epidemiology keywords.
3. **Set Similarity Threshold**: Adjust the threshold to control filtering sensitivity. A higher threshold yields more relevant but fewer results.
4. **Classify and Extract Methods**: The app will display the filtered papers, classify each one by research method, and show specific deep learning methods if mentioned.

## NLP Technique
### Semantic Filtering with TF-IDF and Cosine Similarity
The app uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization combined with cosine similarity for filtering papers:

- **TF-IDF Vectorization**: Converts each paper's combined title and abstract into a weighted vector representation based on word importance. Terms that are common across many papers get lower weights, while unique or important terms are weighted higher.
- **Cosine Similarity**: Measures the similarity between each paper’s TF-IDF vector and a vector representing the target terms. Papers with a similarity score above the threshold are considered relevant.

### Why this approach is effective
This approach goes beyond simple keyword matching:

- **Contextual Relevance**: Unlike basic keyword search, TF-IDF and cosine similarity capture the importance and context of words, allowing for more accurate relevance filtering.
- **Flexibility**: By adjusting the threshold, users can control the level of relevance required, which is useful for refining results based on specific needs.
- **Keyword Independence**: Papers discussing similar topics with different terminology can still be identified as relevant, reducing the chances of missing important studies.

## Approach Justification
This semantic approach is more effective than keywords-based filtering because:

- **Reduces False Positives**: By calculating similarity scores, it can filter out papers that merely mention keywords without actually discussing deep learning applications in virology/epidemiology.
- **Enhances Flexibility**: Allows researchers to tune the threshold, providing a balance between precision and recall based on their specific needs.
- **Captures Context**: TF-IDF and cosine similarity go beyond mere word occurrence, helping identify papers with related concepts even if they use slightly different terminology.
