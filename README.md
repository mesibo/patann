# PatANN - Pattern-Aware Vector Database / ANN

## Overview
PatANN is a massively parallel, distributed, and scalable in-memory/on-disk vector database library for efficient nearest neighbor search across large-scale datasets by finding vector patterns.

PatANN leverages patterns for data partitioning similar to Google ScANN, implements disk-based I/O similar to DiskANN, and employs search techniques like HNSWlib, resulting in an algorithm that combines the best features to outperform existing approaches.

## Status
**Beta Version**: Currently uploaded for benchmarking purposes. Complete documentation and updates are under development. Not for production use yet.

## Platforms
- **Beta Version**: Restricted to Linux to prevent premature circulation of beta versions
- **Production Releases (late Feb 2024)***: Will support all platforms that are supported by mesibo

## Key Features
- Faster Index building and Searching
- Supports both in-memory and on-disk operations
- Dynamic sharding to partition and load balance across servers
- Refined search, filtering and pagination 
- Unlimited scalability without pre-specified capacity

## Algorithmic Approach
- Combines modified NSW (Navigable Small World) graph with a novel pattern based partitioning algorithm
- Preliminary results show phenomenal performance in building index and searching
- Potential slight variations in lower-end matching
- Detailed research paper forthcoming

## Contributions
We are seeking help to:

- Run additional datasets. So far, all tested datasets (including self-generated) exhibit patterns that helps algorithm. We have yet to test datasets without clear patterns or with uniform distribution.
- Validate and improve the algorithm

## Contact
For support / questions, please contact: support@mesibo.com

