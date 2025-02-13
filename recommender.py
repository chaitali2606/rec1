import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from cryptography.fernet import Fernet

# Load dataset
df = pd.read_csv("/Users/chaitalichoudhary/Desktop/JAVA/programs/Project2/venv/filtered_community_data.csv")

# Fill missing values with empty strings and ensure correct types
df["merged_community"] = df["merged_community"].fillna("").astype(str)
df["cleaned_review"] = df["cleaned_review"].fillna("").astype(str)

# Combine community and review text for better recommendations
df["combined_text"] = df["merged_community"] + " " + df["cleaned_review"]

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["combined_text"])

# Compute similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend based on query matching
def get_recommendations(user_query, df, tfidf, cosine_sim):
    # Convert user query into a vector
    query_vector = tfidf.transform([user_query])

    # Compute similarity between query and dataset
    query_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Get top 5 most relevant recommendations (ignore query similarity with itself)
    top_indices = query_similarities.argsort()[-6:-1][::-1]  # Exclude the first index (self-matching)

    # Get recommendations with relevant columns
    recommendations = df.iloc[top_indices][["Clothing ID", "Rating", "composite_score", "sentiment_score", "merged_community", "cleaned_review"]]

    return recommendations
