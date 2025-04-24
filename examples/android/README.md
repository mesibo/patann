# PatANN for Android - Pattern-Aware Vector Database and ANN Framework

## Overview
PatANN is a pattern-aware, massively parallel, and distributed vector search framework designed for scalable and efficient nearest neighbor search, operating both in-memory and on-disk. Unlike conventional algorithms, PatANN leverages macro and micro patterns within vectors to drastically reduce search space before performing costly distance computations.

While still in beta, PatANN's pattern-first approach delivers unprecedented performance advantages. As shown in our benchmarks, PatANN consistently outperforms leading ANN libraries including HNSW (hnswlib), Google ScaNN, Facebook FAISS variants, and others in the critical recall-throughput tradeoff.

![PatANN Benchmark](https://patann.dev/plots_light/sift-128-euclidean.png)

## Android Implementation

This repository contains Android implementation examples:

- **PatANNExampleKotlin**: Synchronous and Asynchronous Android implementation using Kotlin
- **PatANNExampleJava**: Synchronous and Asynchronous Android implementation using Java

## Documentation and Resources

For complete documentation, installation instructions, and additional resources:

- **Main Website**: https://patann.dev
- **Android Tutorial**: https://patann.dev/tutorial/

## Getting Started

Refer to the Android examples directory for implementation guides and the tutorial on the PatANN website for detailed integration instructions.

