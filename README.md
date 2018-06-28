[![Build Status](https://travis-ci.org/ghartoularos/hw2.svg?branch=master)](https://travis-ci.org/ghartoularos/hw2)

## Repository for Algorithms HW2

### Implementation of partitioning and hierarchical clustering algorithms for protein active sites.
Two implementation of clustering algorithms. Functions take in a directory containing active sites, computes a similarity matrix (and subsequent distance matrix), and then clusters them based on distances. Rand index was calculated between the algorithms for every possible cluster number _k_. 

Note the rand_index script require additional packages not explicitly listed in requirements.txt file:
* `scipy.cluster.hierarchy.cut_tree`
* `scipy.misc.comb`
* `itertools`
* `tqdm.tqdm_notebook`
* `matplotlib.pyplot`
