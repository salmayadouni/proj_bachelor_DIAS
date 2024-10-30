import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import random
import time

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Create DataFrames
courses_data = {
    "Course Name": [
        "History of the Atlantic World",
        "Riemannian Geometry",
        "Operating Systems",
        "Food Science",
        "Compilers",
        "Intro to computer science",
    ]
}
skills_data = {"Skill": ["Math", "Computer Science"]}
courses_df = pd.DataFrame(courses_data)
skills_df = pd.DataFrame(skills_data)

# Load a pre-trained sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Warm up the model to reduce initial load time effects
model.encode(["Warm-up"], convert_to_tensor=True)

# Function to perform semantic similarity join
def semantic_join(df1, df2, col1, col2, threshold=0.5):
    # Encode sentences from both dataframes
    sentences1 = df1[col1].tolist()
    sentences2 = df2[col2].tolist()
    
    # Encode on CPU for consistency
    embeddings1 = model.encode(sentences1, convert_to_tensor=True, device='cpu')
    embeddings2 = model.encode(sentences2, convert_to_tensor=True, device='cpu')

    # Compute cosine similarity
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    # Create result dataframe
    results = []
    for i, row1 in df1.iterrows():
        for j, row2 in df2.iterrows():
            if cosine_scores[i][j] > threshold:
                results.append({
                    col1: row1[col1],
                    col2: row2[col2],
                    'similarity': cosine_scores[i][j].item()
                })
    
    return pd.DataFrame(results)

# Function to measure execution time
def time_execution():
    start_time = time.time()
    result = semantic_join(courses_df, skills_df, "Course Name", "Skill", threshold=0.3)
    execution_time = time.time() - start_time
    return execution_time, result

# Run multiple times and calculate average execution time
execution_times = []
for _ in range(5):
    exec_time, result = time_execution()
    execution_times.append(exec_time)
    print(f"Execution time: {exec_time:.4f} seconds")

average_time = sum(execution_times) / len(execution_times)
print(f"\nAverage execution time: {average_time:.4f} seconds")
print("\nResult of the semantic join:\n", result)
