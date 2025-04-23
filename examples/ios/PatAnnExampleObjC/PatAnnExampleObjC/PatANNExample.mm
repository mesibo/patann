#import "PatANNExample.h"

@implementation PatANNExample {
    // Instance variables
    PatANNObjC *_annIndex;
    NSArray<NSArray<NSNumber *> *> *_vectors;
    NSArray<NSNumber *> *_vectorIds;
    NSArray<NSNumber *> *_queryVector;
    NSArray<NSNumber *> *_manualDistances;
    NSArray<NSNumber *> *_topIndices;
    BOOL _testResult;
}

// Constants defined directly in implementation
#define VECTOR_DIM 128
#define NUM_VECTORS 100
#define TOP_K 10
#define SEARCH_RADIUS 100
#define CONSTELLATION_SIZE 16
#define RECALL_THRESHOLD 0.8f
#define ON_DISK_INDEX YES
#define TAG @"PatANNExample"

- (instancetype)init {
    self = [super init];
    if (self) {
        _testResult = NO;
    }
    return self;
}

- (BOOL)initializeTest {
    NSLog(@"%@: Initializing PatANN test...", TAG);
    
    // Create an instance with specified dimensions
    if(!ON_DISK_INDEX) {
        _annIndex = [PatANNObjC createInstance:VECTOR_DIM];
        if (_annIndex == nil) {
            NSLog(@"%@: Failed to create PatANN instance", TAG);
            return NO;
        }
    } else {
        
        _annIndex = [PatANNObjC createOnDiskInstance:VECTOR_DIM path:nil name:@"Demo"];
        
        if (_annIndex == nil) {
            NSLog(@"%@: Failed to create PatANN instance", TAG);
            return NO;
        }
        
        // Note: This demo calculates recall by comparing PatANN results with the manually calculated
        // distances. Since we only use vectors generated in the current session and ignore any previously
        // stored vectors, recall measurements would be incorrect if PatANN uses vectors from previous runs.
        // Therefore, we ensure the index is destroyed upon termination or we exit if index exists and delete
        // index to prepare for the next run.
        [_annIndex destroyIndexOnDelete:YES];
        
        if([_annIndex getIndexSize:YES] > 0) {
            NSLog(@"%@: Index already exists", TAG);
            return NO;
        }
    }
    
    
    @try {
        // Configure the index
        [_annIndex this_is_preproduction_software:YES];
        [_annIndex setDistanceType:PatANNDistanceTypeL2Square];
        [_annIndex setRadius:SEARCH_RADIUS];
        [_annIndex setConstellationSize:CONSTELLATION_SIZE];
        
        // Create random vectors for testing
        _vectors = [self generateRandomVectors:NUM_VECTORS dim:VECTOR_DIM];
        
        // Add vectors to the index
        NSMutableArray<NSNumber *> *ids = [NSMutableArray arrayWithCapacity:_vectors.count];
        for (NSUInteger i = 0; i < _vectors.count; i++) {
            NSUInteger vectorId = [_annIndex addVector:_vectors[i]];
            
            // Refer to the comment above for destroyIndexOnDelete
            if(ON_DISK_INDEX && i == 0 && vectorId > 0) {
                NSLog(@"%@: On-Disk Index already exits", TAG);
                return NO;
            }
            
            if (vectorId < 0) {
                NSLog(@"%@: Failed to add vector at index %lu", TAG, (unsigned long)i);
                return NO;
            }
            [ids addObject:@(vectorId)];
        }
        _vectorIds = [ids copy];
        
        // Generate a query vector - using the first vector with slight modification
        NSMutableArray<NSNumber *> *queryVec = [NSMutableArray arrayWithArray:_vectors[0]];
        // Apply small random changes to make it slightly different
        for (int i = 0; i < 10; i++) {
            NSUInteger pos = arc4random_uniform((uint32_t)queryVec.count);
            float oldValue = [queryVec[pos] floatValue];
            float adjustment = (((float)arc4random() / UINT32_MAX) - 0.5f) * 0.1f;
            queryVec[pos] = @(oldValue + adjustment);
        }
        _queryVector = [queryVec copy];
        
        // Prepare manual distances for evaluation
        NSMutableArray<NSNumber *> *distances = [NSMutableArray arrayWithCapacity:_vectors.count];
        for (NSUInteger i = 0; i < _vectors.count; i++) {
            float dist = [_annIndex distance:_queryVector vector2:_vectors[i]];
            [distances addObject:@(dist)];
        }
        _manualDistances = [distances copy];
        
        // Sort vectors by distance to find top K
        _topIndices = [self findTopK:_manualDistances k:TOP_K];
        
        NSLog(@"%@: Manually calculated top %d:", TAG, TOP_K);
        for (NSUInteger i = 0; i < _topIndices.count; i++) {
            NSUInteger idx = [_topIndices[i] unsignedIntegerValue];
            NSLog(@"%@: Vector ID: %@, Distance: %@", TAG, _vectorIds[idx], _manualDistances[idx]);
        }
        
        return YES;
    } @catch (NSException *exception) {
        NSLog(@"%@: Exception during initialization: %@", TAG, exception.reason);
        return NO;
    }
}

- (void)cleanupTest:(PatANNQueryObjC *)query {    
    if (query != nil) {
        [query destroy];
    }
    
    if (_annIndex != nil) {
        [_annIndex destroy];
        _annIndex = nil;
    }
}

- (PatANNQueryObjC *)createQuerySession {
    PatANNQueryObjC *query = [_annIndex createQuerySession:SEARCH_RADIUS count:TOP_K];
    
    if (query == nil) {
        NSLog(@"%@: Failed to create query session", TAG);
    }
    return query;
}

- (BOOL)processResults:(PatANNQueryObjC *)query {
    @try {
        // Get the results
        NSArray<NSNumber *> *resultIds = [query getResults];
        NSArray<NSNumber *> *resultDistances = [query getResultDists];
        int resultCount = (int)resultIds.count;
        
        NSLog(@"%@: Found %d results", TAG, resultCount);
        
        for (int i = 0; i < resultCount; i++) {
            NSLog(@"%@: Result %d: ID=%@, Distance=%@", TAG, i, resultIds[i], resultDistances[i]);
        }
        
        NSLog(@"%@: PatANN query results:", TAG);
        for (int i = 0; i < resultCount; i++) {
            NSLog(@"%@: Vector ID: %@, Distance: %@", TAG, resultIds[i], resultDistances[i]);
        }
        
        // Calculate recall
        float recall = [self calculateRecall:_topIndices vectorIds:_vectorIds resultIds:resultIds resultCount:resultCount];
        
        NSLog(@"%@: Recall: %.2f%%", TAG, recall * 100);
        
        if (recall >= RECALL_THRESHOLD) {
            NSLog(@"%@: Test passed: Recall of %.2f%% exceeds threshold of %.2f%%",
                  TAG, recall * 100, RECALL_THRESHOLD * 100);
            return YES;
        } else {
            NSLog(@"%@: Test results suboptimal: Recall of %.2f%% is below threshold of %.2f%%",
                  TAG, recall * 100, RECALL_THRESHOLD * 100);
            return NO;
        }
    } @catch (NSException *exception) {
        NSLog(@"%@: Exception during result processing: %@", TAG, exception.reason);
        return NO;
    }
}

#pragma mark - PatANNIndexListener Protocol

-(void) PatANNOnIndexUpdate:(PatANNObjC *)ann indexed:(NSInteger)indexed total:(NSInteger)total {
    
    NSLog(@"%@: Index update: %ld/%ld", TAG, indexed, total);
    
    // If indexing is complete, proceed with query
    if (indexed == total && [ann isIndexReady]) {
        PatANNQueryObjC *query = [self createQuerySession];
        if (query != nil) {
            [query setListener:self];
            [query query:_queryVector k:TOP_K];
        } else {
            _testResult = NO;
            [self cleanupTest:nil];
        }
    }
}

#pragma mark - PatANNQueryListener Protocol

-(void) PatANNOnResult:(PatANNQueryObjC *)query {
    NSLog(@"%@: Query completed", TAG);
    _testResult = [self processResults:query];
    [self cleanupTest:query];
}

#pragma mark - Public Methods

- (BOOL)runTestAsync {
    // Reset results
    _testResult = NO;
    
    // Initialize the test
    if (![self initializeTest]) {
        [self cleanupTest:nil];
        return NO;
    }
    
    // Set the index listener for async notifications
    [_annIndex setIndexListener:self];
    
    // The process continues in PatANN_onIndexUpdate and then patANN_onResult callbacks
    return YES;
}

- (BOOL)runTestSync {
    // Initialize the test
    if (![self initializeTest]) {
        [self cleanupTest:nil];
        return NO;
    }
    
    @try {
        // Wait for the index to be ready (blocking call)
        [_annIndex waitForIndexReady];
        
        if (![_annIndex isIndexReady]) {
            NSLog(@"%@: Index not ready after waiting", TAG);
            [self cleanupTest:nil];
            return NO;
        }
        
        // Create query session
        PatANNQueryObjC *query = [self createQuerySession];
        if (query == nil) {
            [self cleanupTest:nil];
            return NO;
        }
        
        // Perform the search
        [query query:_queryVector k:TOP_K];
        
        // Process the results
        BOOL result = [self processResults:query];
        
        // Cleanup
        [self cleanupTest:query];
        
        return result;
    } @catch (NSException *exception) {
        NSLog(@"%@: Exception occurred: %@", TAG, exception.reason);
        [self cleanupTest:nil];
        return NO;
    }
}

- (BOOL)getAsyncTestResult {
    return _testResult;
}

#pragma mark - Helper Methods

- (NSArray<NSArray<NSNumber *> *> *)generateRandomVectors:(int)count dim:(int)dim {
    NSMutableArray *vectors = [NSMutableArray arrayWithCapacity:count];
    
    // Seed random number generator for reproducibility
    //srand(42);
    
    for (int i = 0; i < count; i++) {
        NSMutableArray *vector = [NSMutableArray arrayWithCapacity:dim];
        for (int j = 0; j < dim; j++) {
            // Values between -1 and 1
            float value = ((float)rand() / RAND_MAX) * 2 - 1;
            [vector addObject:@(value)];
        }
        [vectors addObject:[vector copy]];
    }
    
    return [vectors copy];
}

- (NSArray<NSNumber *> *)findTopK:(NSArray<NSNumber *> *)distances k:(int)k {
    // Create an array of indices
    NSMutableArray *indices = [NSMutableArray arrayWithCapacity:distances.count];
    for (NSUInteger i = 0; i < distances.count; i++) {
        [indices addObject:@(i)];
    }
    
    // Sort indices by their corresponding distances
    [indices sortUsingComparator:^NSComparisonResult(NSNumber *a, NSNumber *b) {
        NSUInteger idxA = [a unsignedIntegerValue];
        NSUInteger idxB = [b unsignedIntegerValue];
        return [distances[idxA] compare:distances[idxB]];
    }];
    
    // Take the top K indices
    int resultSize = MIN(k, (int)distances.count);
    return [indices subarrayWithRange:NSMakeRange(0, resultSize)];
}

- (float)calculateRecall:(NSArray<NSNumber *> *)topIndices 
               vectorIds:(NSArray<NSNumber *> *)vectorIds 
               resultIds:(NSArray<NSNumber *> *)resultIds 
             resultCount:(int)resultCount {
    // Convert topIndices to vectorIds for ground truth
    NSMutableArray *groundTruthIds = [NSMutableArray arrayWithCapacity:topIndices.count];
    for (NSNumber *index in topIndices) {
        NSUInteger idx = [index unsignedIntegerValue];
        [groundTruthIds addObject:vectorIds[idx]];
    }
    
    // Count how many of the ground truth IDs are in the result set
    int matchCount = 0;
    for (int i = 0; i < resultCount; i++) {
        for (NSNumber *groundTruthId in groundTruthIds) {
            if ([resultIds[i] isEqualToNumber:groundTruthId]) {
                matchCount++;
                break;
            }
        }
    }
    
    // Calculate recall as the fraction of ground truth found
    float recall = (float)matchCount / (float)groundTruthIds.count;
    
    NSLog(@"%@: Found %d out of %lu ground truth vectors", TAG, matchCount, (unsigned long)groundTruthIds.count);
    
    return recall;
}

@end
