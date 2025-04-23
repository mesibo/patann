/**
 * PatANNExample - Demonstration of using PatANN for approximate nearest neighbor search
 *
 * This class provides examples of both synchronous and asynchronous usage of PatANN.
 *
 * Usage Patterns:
 *
 * 1. Asynchronous Mode (Recommended for Mobile Applications):
 *    - Call runTestAsync() which returns immediately
 *    - Index building happens in background
 *    - PatANN_onIndexUpdate callback is triggered when indexing is complete
 *    - Query is initiated from the callback
 *    - patANN_onResult callback is triggered when query completes
 *    - Check getAsyncTestResult() to get the final result
 *
 *    This approach avoids blocking the main thread, which is crucial for
 *    mobile applications to prevent ANR (Application Not Responding) errors.
 *
 * 2. Synchronous Mode (Not recommended for mobile UI thread):
 *    - Call runTestSync() which blocks until completion
 *    - waitForIndexReady() blocks until indexing is complete
 *    - Query is performed and results are processed in the same thread
 *    - Function returns the test result
 *
 *    This approach is simpler but can cause ANR if used on the main thread.
 *
 * Common Pattern for Both:
 * - Initialize PatANN
 * - Create an index with a specified dimension
 * - Configure the index (distance type, search radius, etc.)
 * - Add vectors to the index
 * - Create a query session
 * - Perform the search
 * - Evaluate results by comparing with ground truth
 * - Clean up resources
 */
package com.example.patannexamplejava;

import android.content.Context;
import android.util.Log;

import com.mesibo.patann.PatANN;
import com.mesibo.patann.PatANNDistanceType;
import com.mesibo.patann.PatANNIndexListener;
import com.mesibo.patann.PatANNQuery;
import com.mesibo.patann.PatANNQueryListener;
import com.mesibo.patann.PatANNUtils;

import java.util.Arrays;
import java.util.Random;

public class PatANNExample implements PatANNIndexListener, PatANNQueryListener {

    private static final String TAG = "PatANNExample";

    // Configuration parameters as static variables
    private static final int VECTOR_DIM = 128;
    private static final int NUM_VECTORS = 100;
    private static final int TOP_K = 10;
    private static final int SEARCH_RADIUS = 100;
    private static final int CONSTELLATION_SIZE = 16;
    private static final float RECALL_THRESHOLD = 0.8f; // Consider test successful if recall is at least 80%
    private static final boolean ON_DISK_INDEX  = false;

    // Common variables for both sync and async methods
    private PatANN annIndex;
    private float[][] vectors;
    private long[] vectorIds;
    private float[] queryVector;
    private float[] manualDistances;
    private int[] topIndices;
    private boolean testResult = false;


    public PatANNExample(Context context) {
        if (!PatANN.init(context)) {
            Log.e(TAG, "Failed to initialize PatANN");
        }
    }

    /**
     * Common initialization function for both sync and async tests
     */
    private boolean initializeTest() {
        // Create an instance with specified dimensions

        if(!ON_DISK_INDEX) {
            // create In-Memory Index
            annIndex = PatANN.createInstance(VECTOR_DIM);
        } else {
            // create On-Disk Index - null to use default path
            annIndex = PatANN.createOnDiskInstance(VECTOR_DIM, null, "demo");

            // Note: This demo calculates recall by comparing PatANN results with the manually calculated
            // distances. Since we only use vectors generated in the current session and ignore any previously
            // stored vectors, recall measurements would be incorrect if PatANN uses vectors from previous runs.
            // Therefore, we ensure the index is destroyed upon termination or we exit if index exists and delete
            // index to prepare for the next run.
            annIndex.destroyIndexOnDelete(true);

            if(annIndex.getIndexSize(true) > 0) {
                Log.e(TAG, "Index already exists");
                return false;
            }
        }

        if (annIndex == null) {
            Log.e(TAG, "Failed to create PatANN instance");
            return false;
        }

        try {
            // Configure the index
            annIndex.this_is_preproduction_software(true);
            annIndex.setDistanceType(PatANNDistanceType.L2_SQUARE); // Using L2 square distance
            annIndex.setRadius(SEARCH_RADIUS); // Set search radius
            annIndex.setConstellationSize(CONSTELLATION_SIZE); // Set constellation size
            // Let threads be set automatically

            // Create some random vectors for testing
            vectors = generateRandomVectors(NUM_VECTORS, VECTOR_DIM);

            // Add vectors to the index
            vectorIds = new long[vectors.length];
            for (int i = 0; i < vectors.length; i++) {
                vectorIds[i] = annIndex.addVector(vectors[i]);
                if (vectorIds[i] < 0) {
                    Log.e(TAG, "Failed to add vector at index " + i);
                    return false;
                }
            }

            // Generate a query vector - using the first vector with slight modification
            queryVector = Arrays.copyOf(vectors[0], vectors[0].length);
            // Apply small random changes to make it slightly different
            Random random = new Random();
            for (int i = 0; i < 10; i++) {
                int pos = random.nextInt(queryVector.length);
                queryVector[pos] += (random.nextFloat() - 0.5f) * 0.1f;
            }

            // Prepare manual distances for evaluation
            manualDistances = new float[vectors.length];
            for (int i = 0; i < vectors.length; i++) {
                manualDistances[i] = annIndex.distance(queryVector, vectors[i]);
            }

            // Sort vectors by distance to find top K
            topIndices = findTopK(manualDistances, TOP_K);

            Log.d(TAG, "Manually calculated top " + TOP_K + ":");
            for (int i = 0; i < topIndices.length; i++) {
                int idx = topIndices[i];
                Log.d(TAG, "Vector ID: " + vectorIds[idx] + ", Distance: " + manualDistances[idx]);
            }

            return true;
        } catch (Exception e) {
            Log.e(TAG, "Exception during initialization: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    /**
     * Common cleanup function for both sync and async tests
     *
     * @param query The query object to destroy, can be null
     */
    private void cleanupTest(PatANNQuery query) {
        if (query != null) {
            query.destroy();
        }
        if (annIndex != null) {
            annIndex.destroy();
            annIndex = null;
        }
    }

    /**
     * Common function to create query session and prepare for search
     *
     * @return query session object or null if creation failed
     */
    private PatANNQuery createQuerySession() {
        // Create a query session
        PatANNQuery query = annIndex.createQuerySession(SEARCH_RADIUS, TOP_K);

        if (query == null) {
            Log.e(TAG, "Failed to create query session");
        }
        return query;
    }

    /**
     * Common function to process query results and evaluate performance
     *
     * @param query The query object containing the results
     * @return true if the test passed, false otherwise
     */
    private boolean processResults(PatANNQuery query) {
        try {
            // Get the results
            long[] resultIds = query.getResults();
            float[] resultDistances = query.getResultDists();
            int resultCount = resultIds.length;

            Log.d(TAG, "Found " + resultCount + " results");

            for (int i = 0; i < resultCount; i++) {
                Log.d(TAG, "Result " + i + ": ID=" + resultIds[i] +
                        ", Distance=" + resultDistances[i]);
            }

            Log.d(TAG, "PatANN query results:");
            for (int i = 0; i < resultCount; i++) {
                Log.d(TAG, "Vector ID: " + resultIds[i] + ", Distance: " + resultDistances[i]);
            }

            // Calculate recall
            float recall = calculateRecall(topIndices, vectorIds, resultIds, resultCount);

            Log.d(TAG, String.format("Recall: %.2f%%", recall * 100));

            if (recall >= RECALL_THRESHOLD) {
                Log.d(TAG, String.format("Test passed: Recall of %.2f%% exceeds threshold of %.2f%%",
                        recall * 100, RECALL_THRESHOLD * 100));
                return true;
            } else {
                Log.w(TAG, String.format("Test results suboptimal: Recall of %.2f%% is below threshold of %.2f%%",
                        recall * 100, RECALL_THRESHOLD * 100));
                return false;
            }
        } catch (Exception e) {
            Log.e(TAG, "Exception during result processing: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    /**
     * Implementation of PatANNIndexListener interface
     * Called when the index is updated
     */
    @Override
    public void PatANNOnIndexUpdate(PatANN ann, long indexed, long total) {
        Log.d(TAG, "Index update: " + indexed + "/" + total);

        // If indexing is complete, proceed with query
        if (indexed == total && ann.isIndexReady()) {
            PatANNQuery query = createQuerySession();
            if (query != null) {
                query.setListener(this);
                query.query(queryVector, TOP_K);
            } else {
                testResult = false;
                cleanupTest(null);
            }
        }
    }

    /**
     * Implementation of PatANNQueryListener interface
     * Called when a query is completed
     */
    @Override
    public void PatANNOnResult(PatANNQuery query) {
        Log.d(TAG, "Query completed");
        testResult = processResults(query);
        cleanupTest(query);
    }

    /**
     * Run the test using asynchronous API
     * This is the recommended approach for mobile applications as it doesn't block the UI thread
     *
     * Execution flow:
     * 1. Initialize and setup all data
     * 2. Set index listener
     * 3. Return immediately (non-blocking)
     * 4. When index is ready, PatANN_onIndexUpdate callback is triggered
     * 5. Query is initiated from the callback
     * 6. When query completes, patANN_onResult callback is triggered
     * 7. Results are processed and stored in testResult
     *
     * @return true if the test setup was successful (not the final test result)
     */
    public boolean runTestAsync() {
        // Reset results
        testResult = false;

        // Initialize the test
        if (!initializeTest()) {
            cleanupTest(null);
            return false;
        }

        // Set the index listener for async notifications
        annIndex.setIndexListener(this);

        // The process continues in PatANN_onIndexUpdate and then patANN_onResult callbacks
        return true;
    }

    /**
     * Run the test using synchronous API
     * This is not recommended for mobile applications but included for demonstration
     *
     * Execution flow:
     * 1. Initialize and setup all data
     * 2. Wait for index to be ready (BLOCKING CALL)
     * 3. Create query session and run query
     * 4. Process results immediately
     * 5. Cleanup and return result
     *
     * WARNING: This method blocks until completion and should not be used on the UI thread
     * as it may cause ANR (Application Not Responding) errors.
     *
     * @return true if the test passed, false otherwise
     */
    public boolean runTestSync() {
        // Initialize the test
        if (!initializeTest()) {
            cleanupTest(null);
            return false;
        }

        try {
            // Wait for the index to be ready (blocking call)
            annIndex.waitForIndexReady();

            if (!annIndex.isIndexReady()) {
                Log.e(TAG, "Index not ready after waiting");
                cleanupTest(null);
                return false;
            }

            // Create query session
            PatANNQuery query = createQuerySession();
            if (query == null) {
                cleanupTest(null);
                return false;
            }

            // Perform the search
            query.query(queryVector, TOP_K);

            // Process the results
            boolean result = processResults(query);

            // Cleanup
            cleanupTest(query);

            return result;
        } catch (Exception e) {
            Log.e(TAG, "Exception occurred: " + e.getMessage());
            e.printStackTrace();
            cleanupTest(null);
            return false;
        }
    }

    /**
     * Check if the async test has completed and get the result
     * This should be called after runTestAsync() to check the status and result
     *
     * When using async mode, call runTestAsync() to start the test, then periodically
     * call this method to check if the test has completed and get the final result.
     *
     * @return true if the test passed, false otherwise (or if not yet completed)
     */
    public boolean getAsyncTestResult() {
        return testResult;
    }

    /**
     * Generates random vectors for testing
     *
     * @param count Number of vectors to generate
     * @param dim Dimension of each vector
     * @return Array of random vectors
     */
    private static float[][] generateRandomVectors(int count, int dim) {
        float[][] vectors = new float[count][dim];

        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dim; j++) {
                vectors[i][j] = ((float)PatANNUtils.getRandom() * 2) - 1; // Values between -1 and 1
            }
        }

        return vectors;
    }

    /**
     * Finds the indices of the top K smallest elements in an array
     *
     * @param distances Array of distances
     * @param k Number of top elements to find
     * @return Array of indices of top K elements
     */
    private static int[] findTopK(float[] distances, int k) {
        // Create an array of indices
        Integer[] indices = new Integer[distances.length];
        for (int i = 0; i < distances.length; i++) {
            indices[i] = i;
        }

        // Sort indices by their corresponding distances
        Arrays.sort(indices, (a, b) -> Float.compare(distances[a], distances[b]));

        // Take the top K indices
        int[] result = new int[Math.min(k, distances.length)];
        for (int i = 0; i < result.length; i++) {
            result[i] = indices[i];
        }

        return result;
    }

    /**
     * Calculates recall - the fraction of ground truth results found by the ANN search
     *
     * @param topIndices Indices of manually calculated top vectors (ground truth)
     * @param vectorIds IDs of all vectors
     * @param resultIds IDs returned by PatANN query
     * @param resultCount Number of results returned by PatANN
     * @return recall value between 0.0 and 1.0
     */
    private static float calculateRecall(int[] topIndices, long[] vectorIds, long[] resultIds, int resultCount) {
        // Convert topIndices to vectorIds for ground truth
        long[] groundTruthIds = new long[topIndices.length];
        for (int i = 0; i < topIndices.length; i++) {
            groundTruthIds[i] = vectorIds[topIndices[i]];
        }

        // Count how many of the ground truth IDs are in the result set
        int matchCount = 0;
        for (int i = 0; i < resultCount; i++) {
            for (long groundTruthId : groundTruthIds) {
                if (resultIds[i] == groundTruthId) {
                    matchCount++;
                    break;
                }
            }
        }

        // Calculate recall as the fraction of ground truth found
        float recall = (float) matchCount / groundTruthIds.length;

        Log.d(TAG, "Found " + matchCount + " out of " + groundTruthIds.length + " ground truth vectors");

        return recall;
    }
}