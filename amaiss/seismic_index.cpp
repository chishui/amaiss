#include "amaiss/seismic_index.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "amaiss/cluster/inverted_list_clusters.h"
#include "amaiss/cluster/random_kmeans.h"
#include "amaiss/index.h"
#include "amaiss/invlists/inverted_lists.h"
#include "amaiss/io/seismic_invlists_writer.h"
#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"
#include "amaiss/utils/checks.h"
#if defined(__AVX512F__)
#include "amaiss/utils/distance_avx512.h"
#else
#include "amaiss/utils/distance.h"
#endif
#include "absl/container/flat_hash_set.h"
#include "amaiss/utils/prefetch.h"
#include "amaiss/utils/ranker.h"
#include "amaiss/utils/vector_process.h"

namespace amaiss {

static void query_single_inverted_list(
    const SparseVectors* vectors, const InvertedListClusters& cluster_invlist,
    const std::vector<float>& dense, TopKHolder<idx_t>& heap,
    absl::flat_hash_set<idx_t>& visited, float heap_factor, bool first_list) {
    // Skip empty clusters
    size_t csize = cluster_invlist.cluster_size();
    if (csize == 0) {
        return;
    }
    const auto& summaries = cluster_invlist.summaries();
    // compute dp with all summaries
    auto summary_scores = dot_product_float_dense(&summaries, dense);
    size_t num_vectors = vectors->num_vectors();

    std::vector<size_t> cluster_order(summary_scores.size());
    std::iota(cluster_order.begin(), cluster_order.end(), 0);
    if (first_list) {
        std::ranges::sort(cluster_order, [&](size_t a, size_t b) {
            return summary_scores[a] > summary_scores[b];
        });
    }

    const auto& [indptr, indices, values] = vectors->get_all_data();

    for (const size_t& cluster_id : cluster_order) {
        const auto& cluster_score = summary_scores[cluster_id];
        if (heap.full() && (cluster_score * heap_factor < heap.peek_score())) {
            if (first_list) {
                break;
            }
            continue;
        }
        const auto& docs = cluster_invlist.get_docs(cluster_id);
        const size_t n_docs = docs.size();
        for (size_t i = 0; i < n_docs; ++i) {
            const auto& doc_id = docs[i];
            if (i + 1 < n_docs) {
                const idx_t next_doc = docs[i + 1];
                const idx_t next_start = indptr[next_doc];
                const size_t next_len = indptr[next_doc + 1] - next_start;
                prefetch_vector(indices + next_start, values + next_start,
                                next_len);
            }
            auto [_, inserted] = visited.insert(doc_id);
            if (!inserted) {
                continue;
            }
            const idx_t start = indptr[doc_id];
            const size_t len = indptr[doc_id + 1] - start;
            auto score = dot_product_float_dense(
                indices + start, values + start, len, dense.data());
            heap.add(score, doc_id);
        }
    }
}

SeismicIndex::SeismicIndex(int dim)
    : Index(dim), lambda_(0), beta_(0), alpha_(0.4F) {}
SeismicIndex::SeismicIndex(int lambda, int beta, float alpha, int dim)
    : Index(dim), lambda_(lambda), beta_(beta), alpha_(alpha) {}

void SeismicIndex::add(idx_t n, const idx_t* indptr, const term_t* indices,
                       const float* values) {
    throw_if_not_positive(n);
    throw_if_any_null(indptr, indices, values);

    size_t indptr_size = n + 1;
    size_t nnz = indptr[n];  // Total non-zeros
    constexpr int element_size = U32;
    if (vectors_ == nullptr) {
        vectors_ = std::unique_ptr<SparseVectors>(
            new SparseVectors({.element_size = element_size,
                               .dimension = static_cast<size_t>(dimension_)}));
    }
    vectors_->add_vectors(indptr, indptr_size, indices, nnz,
                          reinterpret_cast<const uint8_t*>(values),
                          nnz * element_size);
}

void SeismicIndex::build() {
    ArrayInvertedLists inverted_lists(get_dimension(), ElementSize::U32);
    size_t n_docs = vectors_->num_vectors();

    const auto& indptr_data = vectors_->indptr_data();
    const auto& indices_data = vectors_->indices_data();
    const auto& values_data = vectors_->values_data_float();

    const size_t element_size = vectors_->get_element_size();
    // inverted_lists.add_entry is thread safe
#pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < n_docs; ++i) {
        int start = indptr_data[i];
        int n_tokens = indptr_data[i + 1] - indptr_data[i];
        for (size_t j = start; j < start + n_tokens; ++j) {
            term_t term_id = indices_data[j];
            inverted_lists.add_entry(
                term_id, i, reinterpret_cast<const uint8_t*>(&values_data[j]));
        }
    }

    // TODO: generate lambda and beta
    clustered_inverted_lists.clear();
    clustered_inverted_lists.resize(inverted_lists.size());
#pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < inverted_lists.size(); ++idx) {
        auto& invlist = inverted_lists[idx];
        const auto& doc_ids = invlist.prune_and_keep_doc_ids(lambda_);
        InvertedListClusters inverted_list_clusters(
            RandomKMeans::train(this, doc_ids, beta_));
        inverted_list_clusters.summarize(vectors_.get(), alpha_);
        clustered_inverted_lists[idx] = std::move(inverted_list_clusters);
        invlist.clear();
    }
}

auto SeismicIndex::search(idx_t n, const idx_t* indptr, const term_t* indices,
                          const float* values, int k,
                          const SearchParameters* search_parameters)
    -> std::vector<std::vector<idx_t>> {
    if (vectors_ == nullptr || n == 0) {
        return std::vector<std::vector<idx_t>>(n);
    }
    size_t indptr_size = n + 1;
    size_t nnz = indptr[n];  // Total non-zeros
    // Create query vectors from input
    SparseVectors query_vectors({.element_size = ElementSize::U32,
                                 .dimension = static_cast<size_t>(dimension_)});
    query_vectors.add_vectors(indptr, indptr_size, indices, nnz,
                              reinterpret_cast<const uint8_t*>(values),
                              nnz * U32);
    std::vector<std::vector<idx_t>> results(n);

    const auto* parameters =
        search_parameters != nullptr
            ? dynamic_cast<const SeismicSearchParameters*>(search_parameters)
            : std::make_unique<SeismicSearchParameters>().get();
    // For each query vector
    const auto* query_indptr = query_vectors.indptr_data();
    const auto* query_indices = query_vectors.indices_data();
    const auto* query_values = query_vectors.values_data_float();

    int num_threads = 8;
    int chunk_size = std::max(1, static_cast<int>(n) / (num_threads * 4));
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (idx_t query_idx = 0; query_idx < n; ++query_idx) {
        const auto& dense = query_vectors.get_dense_vector_float(query_idx);
        const idx_t start = query_indptr[query_idx];
        const size_t len = query_indptr[query_idx + 1] - start;
        const auto& cuts = top_k_tokens(
            query_indices + start, query_values + start, len, parameters->cut);
        results[query_idx] =
            std::move(single_query(dense, cuts, k, parameters->heap_factor));
    }

    return results;
}

/**
 * @brief query logic per single query, could be run multi-threaded
 *
 * @param dense
 * @param cuts
 * @param k
 * @param heap_factor
 * @return std::vector<idx_t>
 */
auto SeismicIndex::single_query(const std::vector<float>& dense,
                                const std::vector<term_t>& cuts, int k,
                                float heap_factor) -> std::vector<idx_t> {
    size_t num_docs = vectors_->num_vectors();
    if (num_docs == 0) {
        return {};
    }
    absl::flat_hash_set<idx_t> visited;
    visited.reserve(cuts.size() * 5000);
    TopKHolder<idx_t> holder(k);
    bool first_list = true;
    for (const auto& term : cuts) {
        if (term >= clustered_inverted_lists.size()) [[unlikely]] {
            continue;
        }
        const auto& cluster_invlist = clustered_inverted_lists[term];
        query_single_inverted_list(vectors_.get(), cluster_invlist, dense,
                                   holder, visited, heap_factor, first_list);
        first_list = false;
    }

    return holder.top_k_descending();
}

const SparseVectors* SeismicIndex::get_vectors() const {
    return vectors_.get();
}

void SeismicIndex::write_index(IOWriter* io_writer) {
    // write vectors
    if (vectors_ == nullptr) {
        empty_sparse_vectors.serialize(io_writer);
    } else {
        vectors_->serialize(io_writer);
    }
    SeismicInvertedListsWriter inv_list_writer(clustered_inverted_lists);
    inv_list_writer.serialize(io_writer);
}

void SeismicIndex::read_index(IOReader* io_reader) {
    SparseVectors tmp_vectors;
    tmp_vectors.deserialize(io_reader);
    if (tmp_vectors.num_vectors() > 0) {
        vectors_ = std::make_unique<SparseVectors>(std::move(tmp_vectors));
    }
    SeismicInvertedListsWriter inv_list_writer({});
    inv_list_writer.deserialize(io_reader);
    clustered_inverted_lists = std::move(inv_list_writer.release());
}
}  // namespace amaiss