T1 and T2 refer to two algorithms for clustering, which is a task in unsupervised learning where data points are grouped together into clusters based on their similarity.

T1 refers to the K-Means algorithm, which partitions a dataset into k clusters. It does this by iteratively finding the mean of each cluster, and then assigning each data point to the closest mean.

T2 refers to the Fuzzy K-Means algorithm, which is an extension of the K-Means algorithm. In Fuzzy K-Means, each data point can belong to multiple clusters with different degrees of membership. This allows for overlapping clusters and more flexible partitioning of the data.

Both algorithms were implemented in a Python script in the code shared, where they were tested on a dataset and the performance of both algorithms was compared using the V-Measure, which is a clustering evaluation metric. The results showed that the Fuzzy K-Means algorithm performed better than the K-Means algorithm.

## Problem Statement
Cluster the dataset into `m` clusters using different algorithms and evaluate their performance using v-measure score.

## Methodology
- Load the digits dataset and select examples of classes `0`, `4` and `8`.
- Plot the selected examples to visualize the data.
- Implement the `cluster_and_evaluate` function to cluster the data using different algorithms and compute the v-measure score.
- Train the clustering models with different algorithms, i.e., KMeans and Agglomerative Clustering with three different linkages (`complete`, `average`, and `ward`).
- Evaluate the performance of each algorithm with `v-measure` score.
- Determine the best performing algorithm.
- Evaluate the best algorithm for `m=2,3,4,5` clusters and determine the best number of clusters.
- Compute the contingency matrix and report the measures.

## Results
The best performing algorithm is AgglomerativeClustering with linkage `ward` with `m=3` clusters, with a v-measure score of 0.9740.

## Usage
The code can be run in a Jupyter Notebook or any Python environment by executing the cells in the given order.

## Libraries Used
- numpy
- pandas
- sklearn
- matplotlib
