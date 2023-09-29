# Clustering

This directory hosts implementations of two distributed clustering algorithms, each designed to cluster sequenced DNA data efficiently.

## Table of Contents

- [Q-Gram Clustering](#q-gram-clustering)
- [W-Gram Clustering](#w-gram-clustering)

## Q-Gram Clustering

**Q-Gram clustering** is a distributed clustering algorithm for DNA storage reads proposed in [[1]](#1) that minimizes the number of edit distance calculations by pre-calculating q-gram signatures for each read and then using Hamming distance comparisons between them The implementation follows the paper [[1]](#1), please refer to the paper for more details on the implementation.

## W-Gram Clustering


In **w-gram clustering**, we compute w-gram signatures instead of pre-calculating q-gram signatures. While a q-gram signature signals the existence or non-existence of a set of random substrings within a string, a w-gram signature conveys the position of the specified substrings within the string. We use the L1 norm to measure the dissimilarity between the w-gram signatures. Apart from these two modifications, the algorithm retains its structure from the baseline.



## Usage

The algorithm being used can be picked in makefile and the clustering configurations can be set in clustering_config.cfg. Once they have been set, just start clustering using
```
make run
```

## References
<a id="1">[1]</a> 
Rashtchian, C., Makarychev, K., Racz, M., Ang, S., Jevdjic, D., Yekhanin, S., Ceze, L. and Strauss, K., (2017). **Clustering billions of reads for DNA data storage**. Advances in Neural Information Processing Systems, 30.
