from typing import List, Tuple
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, util
import faiss

# Load a lightweight model for embedding generation
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def generate_embeddings(data: pd.Series, batch_size: int = 100) -> np.ndarray:
    """Generate embeddings for a given pandas Series with batching."""
    embeddings = [
        model.encode(data[i:i + batch_size].tolist(), convert_to_tensor=False, show_progress_bar=False)
        for i in range(0, len(data), batch_size)
    ]
    return np.vstack(embeddings)

def apply_blocking_rule(text1: str, text2: str) -> bool:
    """Simple heuristic blocking rule for initial filtering."""
    return abs(len(text1) - len(text2)) < 10

def sem_join(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    col1: str,
    col2: str,
    join_instruction: str,
    threshold: float = 0.5,
    k: int = 10,
    max_threads: int = 4
) -> pd.DataFrame:
    """
    Perform optimized semantic join with parallel processing and FAISS indexing.
    
    Args:
        df1, df2 (pd.DataFrame): DataFrames to join.
        col1, col2 (str): Columns to join on.
        join_instruction (str): Instruction for the join.
        threshold (float): Cosine similarity threshold.
        k (int): Number of nearest neighbors to retrieve.
        max_threads (int): Max threads for parallel processing.
    
    Returns:
        pd.DataFrame: Joined DataFrame with similarity scores.
    """
    # Generate embeddings for each series
    embeddings1 = generate_embeddings(df1[col1])
    embeddings2 = generate_embeddings(df2[col2])

    # Convert embeddings to lower precision for faster processing
    embeddings1 = embeddings1.astype('float16')
    embeddings2 = embeddings2.astype('float16')

    # Create FAISS index with an IVF index for improved efficiency
    d = embeddings2.shape[1]
    nlist = 100  # Number of clusters
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings2)
    index.add(embeddings2)

    # Function to process a chunk of embeddings
    def process_chunk(start: int, end: int) -> List[Tuple[str, str, float]]:
        results = []
        similarities, indices = index.search(embeddings1[start:end], k)
        for i, (sims, idxs) in enumerate(zip(similarities, indices), start):
            for j, sim in zip(idxs, sims):
                if sim >= threshold and apply_blocking_rule(df1[col1].iloc[i], df2[col2].iloc[j]):
                    results.append((df1[col1].iloc[i], df2[col2].iloc[j], sim))
        return results

    # Run chunks in parallel
    batch_size = len(df1) // max_threads
    with ThreadPoolExecutor(max_threads=max_threads) as executor:
        futures = [
            executor.submit(process_chunk, i, min(i + batch_size, len(df1)))
            for i in range(0, len(df1), batch_size)
        ]
        results = [item for future in futures for item in future.result()]

    return pd.DataFrame(results, columns=[col1, col2, "similarity_score"])

# Sample DataFrames to test the semantic join function
df1 = pd.DataFrame({"Course Name": ["History of the Atlantic World", "Riemannian Geometry", "Operating Systems", "Food Science", "Compilers", "Intro to computer science"]})
df2 = pd.DataFrame({"Skill": ["Math", "Computer Science"]})

# Run the semantic join
result = sem_join(df1, df2, "Course Name", "Skill", "Taking {Course Name} will help me learn {Skill}", threshold=0.3)
print(result)
