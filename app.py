import streamlit as st
import functions  # Importing the helper functions from an external file named 'functions.py'

# Set the title and description for the Streamlit app
st.title("Research Paper Filtering and Classification")
st.write("""
This application filters and classifies research papers based on their relevance to deep learning in virology and epidemiology.
""")

st.subheader("Step 1: Upload Data")

# File uploader for the user to upload a CSV file
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    # Load and display the uploaded data
    df = functions.load_data(uploaded_file)

    # Check if data is loaded successfully and display it
    if df is not None:
        st.write("Dataset loaded successfully!")
        st.dataframe(df)

        st.subheader("Step 2: Filter the papers")

        # Display an informational message about the filtering process
        st.info(
            "Target terms must be selected as cosine similarity will be calculated based on target terms and each paper's abstract and title."
        )

        # Define and display target terms for filtering
        # These are pre-defined keywords relevant to the task
        target_terms = [
            "deep learning", "neural network", "convolutional neural network", "CNN",
            "recurrent neural network", "RNN", "LSTM", "transformer", "epidemiology",
            "virology"
        ]

        # Allow user to select specific target terms for filtering
        target_terms = st.multiselect("Select Target Terms", options=target_terms, default=target_terms)

        # Inform user about the cosine similarity calculation and filtering threshold
        st.info(
            "Cosine similarity will be calculated, and papers will be filtered based on the specified threshold."
        )

        # Slider for setting the similarity threshold for filtering
        threshold = st.slider("Set similarity threshold:", 0.0, 1.0, 0.2)

        # Call the filtering function and filter the DataFrame based on selected terms and threshold
        filtered_df = functions.filter_papers(df, target_terms, threshold)

        # Display the number of matching papers and show the filtered DataFrame
        st.write(f"{len(filtered_df)} papers matched the criteria.")
        st.dataframe(filtered_df[["Title", "similarity_score"]])

        st.subheader("Step 3: Classify and extract methods")

        # Apply classification function to determine the method category for each paper
        filtered_df['method_category'] = filtered_df['processed_text'].apply(functions.classify_method)

        # Apply extraction function to identify specific deep learning methods used in each paper
        filtered_df['methods_used'] = filtered_df['processed_text'].apply(functions.extract_methods)

        # Display the final filtered and classified results
        st.write("Filtered and classified papers:")
        st.dataframe(filtered_df[['Title', 'similarity_score', 'method_category', 'methods_used']])
