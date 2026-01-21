#ifndef KMEANS_UTILS_H
#define KMEANS_UTILS_H

#include <vector>

#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"

namespace amaiss {

void map_docs_to_clusters(const SparseVectors* vectors,
                          const std::vector<idx_t>& docs,
                          std::vector<std::vector<idx_t>>& clusters);
}  // namespace amaiss

#endif  // KMEANS_UTILS_H