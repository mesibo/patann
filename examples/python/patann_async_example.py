#!/usr/bin/python3
"""
PatANN Asynchronous Example

This example demonstrates the asynchronous usage pattern of PatANN:
- Creates a vector index and adds random data vectors
- Sets up asynchronous callbacks for index updates and query results
- Creates a query vector as a modified version of an existing vector
- Executes a query and processes results when they arrive
- Compares results against manually calculated nearest neighbors

This foundational example shows the core asynchronous workflow with a single query.
For multi-query parallel processing, refer to the PatANNAsyncParallelExample.
"""

import time
import patann
import numpy as np
import logging
from patann import PatANNQueryListener, PatANNIndexListener
from patann_utils import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PatANN-Async")

# Listener classes using the wrapper pattern
class IndexListener(PatANNIndexListener):
    def __init__(self, parent=None):
        PatANNIndexListener.__init__(self)
        self.parent = parent

    def PatANNOnIndexUpdate(self, ann, indexed, total):
        return self.parent.on_index_update(ann, indexed, total)

class QueryListener(PatANNQueryListener):
    def __init__(self, parent=None):
        PatANNQueryListener.__init__(self)
        self.parent = parent

    def PatANNOnResult(self, query):
        return self.parent.on_query_result(query)

# Main example class
class PatANNAsyncExample:
    def __init__(self):
        self.ann_index = None    # The PatANN index object
        self.vectors = None      # Reference vectors stored in the index
        self.vector_ids = None   # IDs of vectors in the index
        self.query_vector = None # Vector to search for
        
        self.test_result = False    # Did the test succeed?
        self.test_complete = False  # Is the test finished?
        self.query = None           # Query session object
        
        # Create listener wrappers for callbacks
        self.index_listener = IndexListener(self)
        self.query_listener = QueryListener(self)

    def initialize_test(self):
        """Set up the index and data for testing"""
        try:
            # Create and configure the index with dimension size
            self.ann_index = patann.createInstance(VECTOR_DIM)
            configure_index(self.ann_index)

            # Generate random vectors for testing
            self.vectors = generate_random_vectors(NUM_VECTORS, VECTOR_DIM)

            # Add vectors to the index one by one
            logger.info("Adding vectors to the index...")
            self.vector_ids = []
            for i in range(self.vectors.shape[0]):
                # Ensure proper data type (float32)
                vector = np.ascontiguousarray(self.vectors[i], dtype=np.float32)
                vector_id = self.ann_index.addVector(vector)
                self.vector_ids.append(vector_id)

            # Create a query vector (slightly modified version of an existing vector)
            self.query_vector = create_query_vector(self.vectors)

            return True
            
        except Exception as e:
            logger.error(f"Exception during initialization: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def on_index_update(self, ann, indexed, total):
        """Callback called during index building process"""
        logger.info(f"Index update: {indexed}/{total}")

        # Once indexing is complete, proceed with query
        if indexed == total and ann.isIndexReady():
            logger.info("Index is ready, proceeding with query")
            self.start_query()
        return 0
    
    def start_query(self):
        """Start a search query using the PatANN index"""
        # Create query session with desired search radius and result count
        self.query = self.ann_index.createQuerySession(SEARCH_RADIUS, TOP_K)
        self.query.setListener(self.query_listener, False)
        
        # Ensure the query vector has the proper data type
        query_vector_float32 = np.ascontiguousarray(self.query_vector, dtype=np.float32)
        
        # Execute the query asynchronously
        self.query.query(query_vector_float32, TOP_K)
        logger.info("Started query - results will arrive via callback")
    
    def on_query_result(self, query):
        """Callback when query results are ready"""
        logger.info("Query completed")
        
        # Calculate ground truth for evaluation
        # This calculates distances between query vector and all vectors in dataset
        manual_distances = calculate_manual_distances(self.ann_index, self.query_vector, self.vectors)
        # Find the indices of the closest vectors
        top_indices = find_top_k(manual_distances, TOP_K)
        
        # Show the expected top results from manual calculation
        logger.info(f"Manually calculated top {TOP_K}:")
        for i in range(min(3, len(top_indices))):  # Show just first 3 for brevity
            idx = top_indices[i]
            logger.info(f"Vector ID: {self.vector_ids[idx]}, Distance: {manual_distances[idx]}")
        
        # Compare PatANN results with manual calculation
        result = process_results(query, top_indices, self.vector_ids)
        
        # Update test status
        self.test_result = result
        self.test_complete = True
        
        # Clean up query resources
        query.destroy()
        self.query = None
        
        return 0
    
    def cleanup_test(self):
        """Release resources used by the test"""
        if self.query is not None:
            self.query.destroy()
            self.query = None
            
        if self.ann_index is not None:
            self.ann_index.releaseObject()
            self.ann_index.destroy()
            self.ann_index = None
    
    def run_test_async(self):
        """Run the asynchronous test from start to finish"""
        # Reset results
        self.test_result = False
        self.test_complete = False

        # Initialize the test
        if not self.initialize_test():
            self.cleanup_test()
            return False

        # Set the index listener for async notifications during index building
        self.ann_index.setIndexListener(self.index_listener, 0)
        logger.info("Async test started - index building in background")

        # Note: The process continues in callbacks
        # - on_index_update will be called as index builds
        # - start_query will be called when index is ready
        # - on_query_result will be called when query completes
        return True
    
    def get_test_result(self):
        """Check if the test has completed and get the result"""
        return self.test_complete, self.test_result

# Main function
if __name__ == "__main__":
    try:
        logger.info("Creating PatANNAsyncExample instance")
        example = PatANNAsyncExample()
        logger.info("Instance created successfully")
        
        # Start the async test
        if example.run_test_async():
            logger.info("Async test started successfully")
            
            # Poll for completion - in a real app, this might be event-driven
            while True:
                is_complete, result = example.get_test_result()
                if is_complete:
                    logger.info(f"Async test {'passed' if result else 'failed'}")
                    break
                time.sleep(0.1)
            
            example.cleanup_test()
        else:
            logger.error("Failed to start async test")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        logger.error(traceback.format_exc())
