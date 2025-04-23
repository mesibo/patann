#!/usr/bin/python3
"""
PatANN Synchronous Example - Using PatANN for approximate nearest neighbor search

This example shows the synchronous usage of PatANN:
- Initializes PatANN
- Creates and configures the index
- Adds vectors to the index
- Waits for index to be ready (blocking call)
- Creates a query session and performs the search
- Evaluates results by comparing with ground truth

Note: The synchronous approach is simpler but blocks the thread until completion.
"""

import patann
import numpy as np
import logging
import traceback
from patann_utils import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PatANN-Sync")

def run_test_sync():
    """Run the test using synchronous API"""
    try:
        # Create an instance with specified dimensions
        ann = patann.createInstance(VECTOR_DIM)
        if ann is None:
            logger.error("Failed to create PatANN instance")
            return False

        # Configure the index
        configure_index(ann)

        # Create random vectors for testing
        vectors = generate_random_vectors(NUM_VECTORS, VECTOR_DIM)
        
        # Add vectors to the index one by one, as in the Java example
        logger.info("Adding vectors to the index...")
        vector_ids = []
        for i in range(vectors.shape[0]):
            # Make sure the vector is a proper float32 numpy array
            vector = np.ascontiguousarray(vectors[i], dtype=np.float32)
            vector_id = ann.addVector(vector)
            vector_ids.append(vector_id)

        # Generate a query vector
        query_vector = create_query_vector(vectors)

        # Calculate manual distances for evaluation
        logger.info("Calculating manual distances...")
        manual_distances = calculate_manual_distances(ann, query_vector, vectors)
        top_indices = find_top_k(manual_distances, TOP_K)

        logger.info(f"Manually calculated top {TOP_K}:")
        for i in range(len(top_indices)):
            idx = top_indices[i]
            logger.info(f"Vector ID: {vector_ids[idx]}, Distance: {manual_distances[idx]}")

        # Wait for the index to be ready (blocking call)
        logger.info("Waiting for index to be ready...")
        ann.waitForIndexReady()

        # Create query session and run query
        logger.info("Creating query session and performing search...")
        query = ann.createQuerySession(SEARCH_RADIUS, TOP_K)

        # Make sure query_vector is proper float32 numpy array
        query_vector_float32 = np.ascontiguousarray(query_vector, dtype=np.float32)
        query.query(query_vector_float32, TOP_K)
        
        # Process the results
        result = process_results(query, top_indices, vector_ids)

        # Cleanup
        query.destroy()
        ann.destroy()

        return result

    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    result = run_test_sync()
    print(f"Synchronous test {'passed' if result else 'failed'}")
