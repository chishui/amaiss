#ifndef KMEANS_UTILS_H
#define KMEANS_UTILS_H

#include <vector>

#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"

namespace amaiss {
/**
 * @brief Map each document to its cluster based on the similarity between doc
 * and centroid.
 *
 * @param vectors: full dataset of sparse vectors
 * @param docs: list of documents to be mapped
 * @param clusters: output clusters, i.e., for each cluster, a list of docs. The
 * first element in each cluster is the centroid of the cluster.
 */
void map_docs_to_clusters(const SparseVectors* vectors,
                          const std::vector<idx_t>& docs,
                          std::vector<std::vector<idx_t>>& clusters);
}  // namespace amaiss

#endif  // KMEANS_UTILS_H