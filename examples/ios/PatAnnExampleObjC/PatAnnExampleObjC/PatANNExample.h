#import <Foundation/Foundation.h>
#import <patann/patann.h>

NS_ASSUME_NONNULL_BEGIN

@interface PatANNExample : NSObject <PatANNIndexListenerObjC, PatANNQueryListenerObjC>

- (instancetype)init;
- (BOOL)runTestAsync;
- (BOOL)runTestSync;
- (BOOL)getAsyncTestResult;

@end

NS_ASSUME_NONNULL_END
