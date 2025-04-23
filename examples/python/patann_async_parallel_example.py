#!/usr/bin/python3
"""
PatANN Asynchronous Parallel Example

This example demonstrates advanced usage of PatANN with multiple parallel queries:
- Creates a vector index and adds random data vectors
- Sets up asynchronous callbacks for index updates and query results
- Generates multiple unique query vectors, each with different characteristics
- Executes multiple queries in parallel and processes results as they arrive
- Evaluates each query against its own ground truth calculation

This builds on the concepts from the PatANNAsyncExample, extending it to
handle multiple concurrent query operations efficiently.
"""

import time
import patann
import numpy as np
import logging
import threading
import random
from patann import PatANNQueryListener, PatANNIndexListener
from patann_utils import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PatANN-Async-Parallel")

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
class PatANNAsyncParallelExample:
    def __init__(self):
        self.ann_index = None     # The PatANN index object
        self.vectors = None       # Reference vectors stored in the index
        self.vector_ids = None    # IDs of vectors in the index
        self.query_vectors = []   # Multiple query vectors for parallel searches
        
        self.test_result = False     # Did any test succeed?
        self.test_complete = False   # Are all tests complete?
        self.pending_queries = 0     # Number of queries still running
        self.lock = threading.Lock() # Thread safety for shared variables
        self.queries = []            # Track query objects and their indices
        
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

            # Create multiple query vectors for parallel searches
            self.create_query_vectors(5)  # Create 5 different query vectors

            return True
            
        except Exception as e:
            logger.error(f"Exception during initialization: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def create_query_vectors(self, num_queries):
        """Create multiple unique query vectors for parallel processing"""
        self.query_vectors = []
        for i in range(num_queries):
            # Use different vectors from the dataset as starting points
            # This creates diversity in the query set
            seed_index = i % len(self.vectors)
            query_vector = self.vectors[seed_index].copy()
            
            # Add randomness to make each query unique
            # Each query will have different nearest neighbors
            for _ in range(10):
                pos = random.randint(0, VECTOR_DIM - 1)
                query_vector[pos] += (random.random() - 0.5) * 0.2
                
            self.query_vectors.append(query_vector)
            logger.info(f"Created query vector {i+1} for parallel processing")
    
    def on_index_update(self, ann, indexed, total):
        """Callback called during index building process"""
        logger.info(f"Index update: {indexed}/{total}")

        # Once indexing is complete, proceed with parallel queries
        if indexed == total and ann.isIndexReady():
            logger.info("Index is ready, proceeding with parallel queries")
            self.start_queries()
        return 0
    
    def start_queries(self):
        """Start multiple queries in parallel"""
        # Number of parallel queries equals the number of query vectors
        num_parallel_queries = len(self.query_vectors)
        
        # Thread-safe initialization of pending_queries
        with self.lock:
            self.pending_queries = num_parallel_queries
        
        for i in range(num_parallel_queries):
            # Create query session with desired search radius and result count
            query = self.ann_index.createQuerySession(SEARCH_RADIUS, TOP_K)
            query.setListener(self.query_listener, False)
            
            # Store query along with its index for identification later
            self.queries.append((query, i))
            
            # Ensure the query vector has the proper data type
            query_vector = self.query_vectors[i]
            query_vector_float32 = np.ascontiguousarray(query_vector, dtype=np.float32)
            
            # Execute the query asynchronously
            query.query(query_vector_float32, TOP_K)
            logger.info(f"Started parallel query {i+1}/{num_parallel_queries}")
    
    def on_query_result(self, query):
        """Callback when a query's results are ready"""
        # Find which query this is and get its index
        query_index = None
        for i, (stored_query, idx) in enumerate(self.queries):
            if stored_query == query:
                query_index = idx
                del self.queries[i]
                break
        
        if query_index is None:
            logger.error("Query not found in tracking list!")
            return 0
            
        logger.info(f"Parallel query {query_index+1} completed")
        
        # Get the corresponding query vector
        query_vector = self.query_vectors[query_index]
        
        # Calculate ground truth on-the-fly for this specific query vector
        # Each query has its own set of expected results
        manual_distances = calculate_manual_distances(self.ann_index, query_vector, self.vectors)
        top_indices = find_top_k(manual_distances, TOP_K)
        
        # Show the expected top results from manual calculation
        logger.info(f"Manually calculated top {TOP_K} for query {query_index+1}:")
        for i in range(min(3, len(top_indices))):  # Show just first 3 for brevity
            idx = top_indices[i]
            logger.info(f"Vector ID: {self.vector_ids[idx]}, Distance: {manual_distances[idx]}")
        
        # Compare PatANN results with manual calculation
        result = process_results(query, top_indices, self.vector_ids)
        
        # Thread-safe update of shared variables
        with self.lock:
            # Update result - consider test passed if any query succeeds
            if result:
                self.test_result = True
                
            self.pending_queries -= 1
            pending = self.pending_queries
            logger.info(f"Remaining queries: {pending}")
            
            # Mark test complete if all queries are done
            if pending == 0:
                self.test_complete = True
        
        # Clean up query resources
        query.destroy()
        
        return 0
    
    def cleanup_test(self):
        """Release resources used by the test"""
        # Clean up any remaining queries
        for query, _ in self.queries:
            query.destroy()
        self.queries = []
        
        # Clean up the index
        if self.ann_index is not None:
            self.ann_index.releaseObject()
            self.ann_index.destroy()
            self.ann_index = None
    
    def run_test_async(self):
        """Run the asynchronous parallel test from start to finish"""
        # Reset results
        self.test_result = False
        self.test_complete = False
        self.pending_queries = 0
        self.queries = []

        # Initialize the test
        if not self.initialize_test():
            self.cleanup_test()
            return False

        # Set the index listener for async notifications during index building
        self.ann_index.setIndexListener(self.index_listener, 0)
        logger.info("Async parallel test started - index building in background")

        # Note: The process continues in callbacks
        # - on_index_update will be called as index builds
        # - start_queries will be called when index is ready
        # - on_query_result will be called as each query completes
        return True
    
    def get_test_result(self):
        """Check if the test has completed and get the result"""
        return self.test_complete, self.test_result

# Main function
if __name__ == "__main__":
    try:
        logger.info("Creating PatANNAsyncParallelExample instance")
        example = PatANNAsyncParallelExample()
        logger.info("Instance created successfully")
        
        # Start the async test
        if example.run_test_async():
            logger.info("Async parallel test started successfully")
            
            # Poll for completion - in a real app, this might be event-driven
            while True:
                is_complete, result = example.get_test_result()
                if is_complete:
                    logger.info(f"Async parallel test {'passed' if result else 'failed'}")
                    break
                time.sleep(0.1)
            
            example.cleanup_test()
        else:
            logger.error("Failed to start async parallel test")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        logger.error(traceback.format_exc())
