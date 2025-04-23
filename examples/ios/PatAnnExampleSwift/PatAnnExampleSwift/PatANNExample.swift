import Foundation

/**
 * PatANNExample - Swift implementation for demonstrating PatANN usage
 *
 * This class provides examples of both synchronous and asynchronous
 * usage of PatANN for approximate nearest neighbor search.
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
 *    mobile applications to prevent freezing.
 *
 * 2. Synchronous Mode (Not recommended for mobile UI thread):
 *    - Call runTestSync() which blocks until completion
 *    - waitForIndexReady() blocks until indexing is complete
 *    - Query is performed and results are processed in the same thread
 *    - Function returns the test result
 *
 *    This approach is simpler but can freeze the UI if used on the main thread.
 */
import Foundation

/**
 * PatANNExample - Swift implementation for demonstrating PatANN usage
 *
 * This class provides examples of both synchronous and asynchronous
 * usage of PatANN for approximate nearest neighbor search.
 */
import Foundation

import patann


/**
 * PatANNExample - Swift implementation for demonstrating PatANN usage
 *
 * This class provides examples of both synchronous and asynchronous
 * usage of PatANN for approximate nearest neighbor search.
 */
class PatANNExample: NSObject, PatANNIndexListener, PatANNQueryListener {
    // MARK: - Constants
    
    // Tag for logging
    private let TAG = "PatANNExample"
    
    // Configuration parameters
    private let VECTOR_DIM: Int = 128
    private let NUM_VECTORS = 100
    private let TOP_K: Int = 10
    private let SEARCH_RADIUS: Int = 100
    private let CONSTELLATION_SIZE: Int = 16
    private let RECALL_THRESHOLD: Float = 0.8 // Consider test successful if recall is at least 80%
    private let ON_DISK_INDEX: Bool = false
    
    // MARK: - Variables
    
    // Common variables for both sync and async methods
    private var annIndex: PatANN?
    private var vectors: [[Float]] = []
    private var vectorIds: [NSNumber] = []
    private var queryVector: [NSNumber] = []
    private var manualDistances: [Float] = []
    private var topIndices: [Int] = []
    private var testResult: Bool = false
    
    // MARK: - Initialization
    
    override init() {
        super.init()
        testResult = false
    }
    
    // MARK: - Test Setup and Cleanup
    
    /**
     * Common initialization function for both sync and async tests
     */
    private func initializeTest() -> Bool {
        print("\(TAG): Initializing PatANN test...")
        
        if(!ON_DISK_INDEX) {
            let index = PatANN.createInstance(VECTOR_DIM)
            annIndex = index
        } else {
            let index = PatANN.createOnDiskInstance(VECTOR_DIM, path: nil, name: "Demo")
            
            // Note: This demo calculates recall by comparing PatANN results with the manually calculated
            // distances. Since we only use vectors generated in the current session and ignore any previously
            // stored vectors, recall measurements would be incorrect if PatANN uses vectors from previous runs.
            // Therefore, we ensure the index is destroyed upon termination or we exit if index exists and delete
            // index to prepare for the next run.
            index.destroyIndexOnDelete(true)
            
            if(index.getIndexSize(true) > 0) {
                print("\(TAG): Index already exists")
                return false;
            }
            
            annIndex = index
        }
        
        // Configure the index
        annIndex?.this_is_preproduction_software(true)
        annIndex?.setDistanceType(.l2Square) // Using L2 square distance
        annIndex?.setRadius(SEARCH_RADIUS)
        annIndex?.setConstellationSize(CONSTELLATION_SIZE)
        // Let threads be set automatically
        
        // Create some random vectors for testing
        vectors = generateRandomVectors(count: NUM_VECTORS, dim: Int(VECTOR_DIM))
        
        // Add vectors to the index
        var ids: [NSNumber] = []
        for vector in vectors {
            // Convert Swift array to NSArray of NSNumbers
            let nsVector = vector.map { NSNumber(value: $0) }
            
            let vectorId = annIndex?.addVector(nsVector) ?? 0
            ids.append(NSNumber(value: vectorId))
        }
        vectorIds = ids
        
        // Generate a query vector - using the first vector with slight modification
        var queryVec = vectors[0]
        
        // Apply small random changes to make it slightly different
        for _ in 0..<10 {
            let pos = Int.random(in: 0..<queryVec.count)
            let adjustment: Float = (Float.random(in: 0...1) - 0.5) * 0.1
            queryVec[pos] += adjustment
        }
        
        // Convert to NSArray of NSNumbers for the API
        queryVector = queryVec.map { NSNumber(value: $0) }
        
        // Prepare manual distances for evaluation
        manualDistances = []
        for vector in vectors {
            let nsVector = vector.map { NSNumber(value: $0) }
            let distance = annIndex?.distance(queryVector, vector2: nsVector) ?? Float.infinity
            manualDistances.append(distance)
        }
        
        // Sort vectors by distance to find top K
        topIndices = findTopK(distances: manualDistances, k: Int(TOP_K))
        
        print("\(TAG): Manually calculated top \(TOP_K):")
        for i in 0..<topIndices.count {
            let idx = topIndices[i]
            print("\(TAG): Vector ID: \(vectorIds[idx]), Distance: \(manualDistances[idx])")
        }
        
        return true
    }
    
    /**
     * Common cleanup function for both sync and async tests
     */
    private func cleanupTest(_ query: PatANNQuery?) {
        if let query = query {
            query.destroy()
        }
        if let index = annIndex {
            index.destroy()
            annIndex = nil
        }
    }
    
    /**
     * Creates a query session for searching vectors
     */
    private func createQuerySession() -> PatANNQuery? {
        guard let index = annIndex else {
            print("\(TAG): Index not available")
            return nil
        }
        
        let query = index.createQuerySession(SEARCH_RADIUS, count: TOP_K)
        
        return query
    }
    
    /**
     * Processes query results and evaluates performance
     */
    private func processResults(_ query: PatANNQuery) -> Bool {
        // Get the results
        let resultIds = query.getResults()
        let resultDistances = query.getResultDists()
        let resultCount = resultIds.count
        
        print("\(TAG): Found \(resultCount) results")
        
        for i in 0..<resultCount {
            print("\(TAG): Result \(i): ID=\(resultIds[i]), Distance=\(resultDistances[i])")
        }
        
        print("\(TAG): PatANN query results:")
        for i in 0..<resultCount {
            print("\(TAG): Vector ID: \(resultIds[i]), Distance: \(resultDistances[i])")
        }
        
        // Calculate recall
        let recall = calculateRecall(
            topIndices: topIndices,
            vectorIds: vectorIds,
            resultIds: resultIds,
            resultCount: resultCount
        )
        
        print("\(TAG): Recall: \(recall * 100)%")
        
        if recall >= RECALL_THRESHOLD {
            print("\(TAG): Test passed: Recall of \(recall * 100)% exceeds threshold of \(RECALL_THRESHOLD * 100)%")
            return true
        } else {
            print("\(TAG): Test results suboptimal: Recall of \(recall * 100)% is below threshold of \(RECALL_THRESHOLD * 100)%")
            return false
        }
    }
    
    // MARK: - PatANN Listener Protocol Implementation
    
    
    /**
     * Implementation of PatANNIndexListener protocol
     * Called when the index is updated
     */
    func PatANNOnIndexUpdate(ann: PatANN, indexed: Int, total: Int) {
        print("\(TAG): Index update: \(indexed)/\(total)")
        
        // If indexing is complete, proceed with query
        if indexed == total && ann.isIndexReady() {
            if let query = createQuerySession() {
                query.setListener(self)
                query.query(queryVector, k: TOP_K)
            } else {
                testResult = false
                cleanupTest(nil)
            }
        }
    }
    
    /**
     * Implementation of PatANNQueryListener protocol
     * Called when a query is completed
     */
 
    func PatANNOnResult(query: PatANNQuery) {
        print("\(TAG): Query completed")
        testResult = processResults(query)
        cleanupTest(query)
    }
    
    // MARK: - Public Test Methods
    
    /**
     * Run the test using asynchronous API
     */
    func runTestAsync() -> Bool {
        // Reset results
        testResult = false
        
        // Initialize the test
        if !initializeTest() {
            cleanupTest(nil)
            return false
        }
        
        // Set the index listener for async notifications
        annIndex?.setIndexListener(self)
        
        // The process continues in PatANN_onIndexUpdate and then patANN_onResult callbacks
        return true
    }
    
    /**
     * Run the test using synchronous API
     */
    func runTestSync() -> Bool {
        // Initialize the test
        if !initializeTest() {
            cleanupTest(nil)
            return false
        }
        
        // Wait for the index to be ready (blocking call)
        annIndex?.waitForIndexReady()
        
        guard let index = annIndex, index.isIndexReady() else {
            print("\(TAG): Index not ready after waiting")
            cleanupTest(nil)
            return false
        }
        
        // Create query session
        guard let query = createQuerySession() else {
            cleanupTest(nil)
            return false
        }
        
        // Perform the search
        query.query(queryVector, k: TOP_K)
        
        // Process the results
        let result = processResults(query)
        
        // Cleanup
        cleanupTest(query)
        
        return result
    }
    
    /**
     * Check if the async test has completed and get the result
     */
    func getAsyncTestResult() -> Bool {
        return testResult
    }
    
    // MARK: - Utility Methods
    
    /**
     * Generates random vectors for testing
     */
    private func generateRandomVectors(count: Int, dim: Int) -> [[Float]] {
        // Seed random number generator for reproducibility
        //srand48(42)
        
        var vectors: [[Float]] = []
        
        for _ in 0..<count {
            var vector: [Float] = []
            for _ in 0..<dim {
                // Values between -1 and 1
                let value = Float(drand48() * 2 - 1)
                vector.append(value)
            }
            vectors.append(vector)
        }
        
        return vectors
    }
    
    /**
     * Finds the indices of the top K smallest elements in an array
     */
    private func findTopK(distances: [Float], k: Int) -> [Int] {
        // Create an array of indices
        let indices = Array(0..<distances.count)
        
        // Sort indices by their corresponding distances
        let sortedIndices = indices.sorted { distances[$0] < distances[$1] }
        
        // Take the top K indices
        let resultSize = min(k, distances.count)
        return Array(sortedIndices.prefix(resultSize))
    }
    
    /**
     * Calculates recall - the fraction of ground truth results found by the ANN search
     */
    private func calculateRecall(
        topIndices: [Int],
        vectorIds: [NSNumber],
        resultIds: [NSNumber],
        resultCount: Int
    ) -> Float {
        // Convert topIndices to vectorIds for ground truth
        var groundTruthIds: [NSNumber] = []
        for index in topIndices {
            groundTruthIds.append(vectorIds[index])
        }
        
        // Count how many of the ground truth IDs are in the result set
        var matchCount = 0
        for i in 0..<resultCount {
            for groundTruthId in groundTruthIds {
                if resultIds[i].isEqual(to: groundTruthId) {
                    matchCount += 1
                    break
                }
            }
        }
        
        // Calculate recall as the fraction of ground truth found
        let recall = Float(matchCount) / Float(groundTruthIds.count)
        
        print("\(TAG): Found \(matchCount) out of \(groundTruthIds.count) ground truth vectors")
        
        return recall
    }
}
