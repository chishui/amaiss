#include "amaiss/cluster/kmeans_utils.h"

#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"
#include "amaiss/utils/dense_vector_matrix.h"

#if defined(__AVX512F__)
#include "amaiss/utils/distance_avx512.h"
#else
#include "amaiss/utils/distance.h"
#endif

namespace amaiss {
#if defined(__AVX512F__)

static std::unique_ptr<DenseVectorMatrix> initialize_cluster_representatives(
    const std::vector<std::vector<float>>& dense_centroids,
    size_t center_dimension) {
    size_t cluster_count = dense_centroids.size();
    auto representatives = std::make_unique<DenseVectorMatrix>(
        center_dimension,  // Number of dimensions
        cluster_count      // Number of clusters
    );

    // Fill the representatives matrix
    float* data = representatives->data();
    for (size_t dim = 0; dim < center_dimension; ++dim) {
        for (size_t cluster_idx = 0; cluster_idx < cluster_count;
             ++cluster_idx) {
            data[dim * cluster_count + cluster_idx] =
                dense_centroids[cluster_idx][dim];
        }
    }

    return representatives;
}

static std::vector<std::vector<float>> centroids_to_dense(
    const std::vector<std::vector<idx_t>>& clusters,
    const SparseVectors* vectors) {
    std::vector<std::vector<float>> dense_centroids;
    dense_centroids.reserve(clusters.size());
    for (const auto& cluster : clusters) {
        const auto& dense = vectors->get_dense_vector_float(cluster.at(0));
        dense_centroids.emplace_back(dense);
    }
    return dense_centroids;
}

static size_t get_dense_vector_max_dimension(
    const std::vector<std::vector<float>>& dense) {
    size_t max_dimension = 0;
    for (const auto& centroid : dense) {
        max_dimension = std::max<size_t>(max_dimension, centroid.size());
    }
    return max_dimension;
}

static void map_docs_to_clusters_avx512(
    const SparseVectors* vectors, const std::vector<idx_t>& docs,
    std::vector<std::vector<idx_t>>& clusters) {
    if (vectors == nullptr) {
        throw std::runtime_error("vectors is nullptr");
    }
    size_t n_clusters = clusters.size();
    size_t n_docs = docs.size();

    auto dense_centroids = centroids_to_dense(clusters, vectors);
    size_t max_dimension = vectors->get_dimension() == 0
                               ? get_dense_vector_max_dimension(dense_centroids)
                               : vectors->get_dimension();
    size_t center_dimension = (n_clusters > 0) ? max_dimension : 0;
    auto cluster_representatives =
        initialize_cluster_representatives(dense_centroids, center_dimension);
    // release memory
    dense_centroids = std::vector<std::vector<float>>();
    for (size_t i = 0; i < n_docs; ++i) {
        idx_t doc_id = docs[i];
        const auto& [indices, weights] = vectors->get_vector_view(doc_id);
        auto similarities = dot_product_sparse_matrix(indices, weights,
                                                      *cluster_representatives);

        size_t best_cluster = argmax_simd(similarities);
        clusters[best_cluster].push_back(doc_id);
    }
}

#endif

void map_docs_to_clusters(const SparseVectors* vectors,
                          const std::vector<idx_t>& docs,
                          std::vector<std::vector<idx_t>>& clusters) {
#if defined(__AVX512F__)
    map_docs_to_clusters_avx512(vectors, docs, clusters);
    return;
#else
    if (vectors == nullptr) {
        throw std::runtime_error("vectors is nullptr");
    }
    size_t n_clusters = clusters.size();
    size_t n_docs = docs.size();
    for (size_t i = 0; i < n_docs; ++i) {
        const auto& vec = vectors->get_dense_vector_float(docs[i]);
        float max_similarity = std::numeric_limits<float>::lowest();
        size_t best_cluster = 0;
        for (size_t j = 0; j < n_clusters; ++j) {
            const auto& [indices, weights] =
                vectors->get_vector_view(clusters[j].at(0));
            const auto& similarity =
                dot_product_float_dense(indices, weights, vec);
            if (similarity > max_similarity) {
                max_similarity = similarity;
                best_cluster = j;
            }
        }
        clusters[best_cluster].push_back(docs[i]);
    }
#endif
}

}  // namespace amaiss