# PatANN - Pattern-Aware Vector Database and ANN Framework

## Overview
PatANN is a pattern-aware, massively parallel, and distributed vector search framework designed for scalable and efficient nearest neighbor search, operating both in-memory and on-disk. Unlike conventional algorithms, PatANN leverages macro and micro patterns within vectors to drastically reduce search space before performing costly distance computations.

Refer to the website for technical details, algorithm overview, key innovations, benchmarks, and tutorials.  

https://patann.dev

While still in beta, PatANN's pattern-first approach delivers unprecedented performance advantages. As shown in our benchmarks (Figure 1), PatANN consistently outperforms leading ANN libraries including HNSW (hnswlib), Google ScaNN, Facebook FAISS variants, and others in the critical recall-throughput tradeoff.

![PatANN Benchmark](https://patann.dev/plots_light/sift-128-euclidean.png)

## Repository Structure

This repository contains:

- **ann-benchmarks**: Benchmarking tools and results comparing PatANN to other ANN libraries. Refer to the README in ann-benchmarks folder for more details
- **examples**: Sample code demonstrating PatANN integration across multiple platforms

### Examples
The examples directory includes implementation samples for multiple platforms. Refer to the tutorial on PatANN website https://patann.dev for details

#### Python (Linux, macOS, Windows)
- `patann_sync_example.py`: Synchronous PatANN API usage example
- `patann_async_example.py`: Asynchronous PatANN API usage example
- `patann_async_parallel_example.py`: Example demonstrating parallel asynchronous vector search
- `patann_utils.py`: Common utility functions used in all examples

#### Android
- `PatANNExampleKotlin`: Synchronous and Asynchronous Android implementation using Kotlin
- `PatANNExampleJava`: Synchronous and Asynchronous Android implementation using Java

#### iOS
- `PatAnnExampleSwift`: Synchronous and Asynchronous iOS implementation using Swift
- `PatAnnExampleObjC`: Synchronous and Asynchronous iOS implementation using Objective-C

## Getting Started

Refer to the examples directory for platform-specific integration guides. Visit our [website](https://patann.dev) for complete documentation, installation instructions, and additional resources.
