#!/usr/bin/python3
"""
PatANN Utils - Common utility functions for PatANN examples
"""

import numpy as np
import random
import logging
import patann  # Add import for patann module

# Configuration parameters
VECTOR_DIM = 128
NUM_VECTORS = 100
TOP_K = 10
SEARCH_RADIUS = 100
CONSTELLATION_SIZE = 16
RECALL_THRESHOLD = 0.8  # Consider test successful if recall is at least 80%

logger = logging.getLogger("PatANN-Utils")

def configure_index(ann):
    """Configure the PatANN index with standard settings"""
    ann.this_is_preproduction_software(True)
    
    # Handle distance type - use direct value (2) for L2_SQUARE distance
    # to avoid any dependency on enum access
    ann.setDistanceType(2)  # L2_SQUARE distance
    ann.setRadius(SEARCH_RADIUS)
    ann.setConstellationSize(CONSTELLATION_SIZE)

def generate_random_vectors(count, dim):
    """Generate random vectors for testing"""
    np.random.seed(42)  # Use fixed seed for reproducibility
    return np.float32(np.random.random((count, dim)) * 2 - 1)  # Values between -1 and 1

def create_query_vector(vectors):
    """Create a query vector based on a slight modification of the first vector"""
    query_vector = vectors[0].copy()
    for _ in range(10):
        pos = random.randint(0, VECTOR_DIM - 1)
        query_vector[pos] += (random.random() - 0.5) * 0.1
    return query_vector

def find_top_k(distances, k):
    """Find the indices of the top K smallest elements in an array"""
    indices = np.argsort(distances)
    return indices[:min(k, len(distances))]

def calculate_manual_distances(ann, query_vector, vectors):
    """Calculate manual distances between query vector and all other vectors"""
    distances = []
    for i in range(len(vectors)):
        # Skip explicit distance calculation and use numpy directly
        # This avoids C++ type conversion issues with the distance function
        v1 = np.array(query_vector, dtype=np.float32)
        v2 = np.array(vectors[i], dtype=np.float32)
        
        # Calculate L2 squared distance directly with numpy
        dist = np.sum((v1 - v2) ** 2)
        distances.append(dist)
    return distances

def calculate_recall(top_indices, vector_ids, result_ids):
    """Calculate recall - the fraction of ground truth results found by the ANN search"""
    ground_truth_ids = [vector_ids[idx] for idx in top_indices]
    
    match_count = 0
    for result_id in result_ids:
        if result_id in ground_truth_ids:
            match_count += 1
    
    recall = match_count / len(ground_truth_ids)
    logger.info(f"Found {match_count} out of {len(ground_truth_ids)} ground truth vectors")
    return recall

def process_results(query, top_indices, vector_ids):
    """Process query results and evaluate performance"""
    # Based on sample.py, getResults takes a parameter (0)
    result_ids = query.getResults(0)
    result_distances = query.getResultDists()
    
    logger.info(f"Found {len(result_ids)} results")
    for i in range(len(result_ids)):
        logger.info(f"Result {i}: ID={result_ids[i]}, Distance={result_distances[i]}")
    
    # Calculate recall
    recall = calculate_recall(top_indices, vector_ids, result_ids)
    logger.info(f"Recall: {recall * 100:.2f}%")
    
    if recall >= RECALL_THRESHOLD:
        logger.info(f"Test passed: Recall of {recall * 100:.2f}% exceeds threshold")
        return True
    else:
        logger.warning(f"Test results suboptimal: Recall of {recall * 100:.2f}% is below threshold")
        return False
