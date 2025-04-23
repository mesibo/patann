//
//  ViewController.m
//  PatAnnExampleObjC


#import "ViewController.h"
#import "patann/patann.h"
#import "PatANNExample.h"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];

    
    BOOL runAsync = YES;
    
    PatANNExample *example = [PatANNExample new];
    
    if(runAsync) {
        [example runTestAsync];
    } else {
        [example runTestSync];
    }
}


@end
