#!/usr/bin/env python3
"""
Example usage of the AMAISS Python bindings.
"""

import numpy as np
import amaiss


def main():
    # Create a BrutalIndex instance
    index = amaiss.BrutalIndex()

    # Example: Add sparse vectors to the index
    # Format: CSR (Compressed Sparse Row) format
    # indptr: pointer array indicating where each vector starts/ends
    # indices: term indices for non-zero values
    # values: corresponding weight values

    # Example: 3 sparse vectors
    # Vector 0: {0: 1.0, 2: 0.5}
    # Vector 1: {1: 2.0, 3: 1.5}
    # Vector 2: {0: 0.8, 1: 1.2, 2: 0.3}

    n_vectors = 3
    indptr = np.array([0, 2, 4, 7], dtype=np.int64)
    indices = np.array([0, 2, 1, 3, 0, 1, 2], dtype=np.uint16)
    values = np.array([1.0, 0.5, 2.0, 1.5, 0.8, 1.2, 0.3], dtype=np.float32)

    # Add vectors to the index
    index.add(n_vectors, indptr, indices, values)
    print(f"Added {n_vectors} vectors to the index")

    # Search for similar vectors
    # Query vector: {0: 1.0, 1: 0.5}
    n_queries = 1
    query_indptr = np.array([0, 2], dtype=np.int64)
    query_indices = np.array([0, 1], dtype=np.uint16)
    query_values = np.array([1.0, 0.5], dtype=np.float32)

    k = 2  # Find top 2 nearest neighbors

    # Perform search - returns a 2D array of shape (n_queries, k)
    labels = index.search(n_queries, query_indptr, query_indices, query_values, k)

    print(f"\nTop {k} nearest neighbors:")
    print(f"Labels shape: {labels.shape}")  # Should be (1, 2)
    for i in range(n_queries):
        neighbors = labels[i]  # Each row is the k neighbors for query i
        print(f"Query {i}: {neighbors}")


if __name__ == "__main__":
    main()
