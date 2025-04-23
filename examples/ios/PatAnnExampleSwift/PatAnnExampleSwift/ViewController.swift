import UIKit

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        let example = PatANNExample()
            
            // Run synchronous test
        print("Running synchronous test...")
        let syncResult = example.runTestAsync()
        print("Synchronous test \(syncResult ? "passed!" : "failed.")")
    }


}

