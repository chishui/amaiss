#include "amaiss/seismic_index.h"

#include <sys/select.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>

#include "amaiss/cluster/inverted_list_clusters.h"
#include "amaiss/cluster/random_kmeans.h"
#include "amaiss/index.h"
#include "amaiss/invlists/inverted_lists.h"
#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"
#include "amaiss/utils/checks.h"
#if defined(__AVX512F__)
#include "amaiss/utils/distance_avx512.h"
#else
#include "amaiss/utils/distance.h"
#endif
#include "amaiss/utils/prefetch.h"
#include "amaiss/utils/ranker.h"
#include "amaiss/utils/vector_process.h"

namespace amaiss {

static void query_single_inverted_list(
    const SparseVectors* vectors, const InvertedListClusters& cluster_invlist,
    const std::vector<float>& dense, TopKHolder<idx_t>& heap,
    std::unordered_set<idx_t>& visited, float heap_factor, bool first_list) {
    // Skip empty clusters
    size_t csize = cluster_invlist.cluster_size();
    if (csize == 0) {
        return;
    }
    const auto& summaries = cluster_invlist.summaries();
    // compute dp with all summaries
    auto summary_scores = dot_product_float_dense(&summaries, dense);
    size_t num_vectors = vectors->num_vectors();
    std::vector<std::pair<size_t, float>> cluster_score_pairs;
    cluster_score_pairs.reserve(summary_scores.size());

    for (size_t i = 0; i < summary_scores.size(); ++i) {
        cluster_score_pairs.emplace_back(i, summary_scores[i]);
    }
    // sort first list to handle higher impact cluster first to avoid bias
    if (first_list) {
        std::ranges::sort(
            cluster_score_pairs,
            [](const auto& a, const auto& b) { return a.second > b.second; });
    }
    for (const auto& [cluster_id, cluster_score] : cluster_score_pairs) {
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
                prefetch_next_vector(vectors, docs[i + 1]);
            }
            auto [_, inserted] = visited.insert(doc_id);
            if (!inserted) {
                continue;
            }
            const auto& [indices, weights] = vectors->get_vector_view(doc_id);
            auto score = dot_product_float_dense(indices, weights, dense);
            heap.add(score, doc_id);
        }
    }
}

SeismicIndex::SeismicIndex(int dim)
    : Index(dim), lambda_(0), beta_(0), alpha_(0.4F) {}
SeismicIndex::SeismicIndex(int lambda, int beta, float alpha, int dim)
    : Index(dim), lambda_(lambda), beta_(beta), alpha_(alpha) {}

void SeismicIndex::add(idx_t n, std::vector<idx_t>& indptr,
                       std::vector<term_t>& indices,
                       std::vector<float>& values) {
    if (vectors_ == nullptr) {
        vectors_ = std::unique_ptr<SparseVectors>(
            new SparseVectors({.element_size = U32,
                               .dimension = static_cast<size_t>(dimension_)}));
    }
    vectors_->add_vectors(indptr, indices, values);
}

void SeismicIndex::build() {
    ArrayInvertedLists inverted_lists(get_dimension(), ElementSize::U32);
    size_t n_docs = vectors_->num_vectors();
    // inverted_lists.add_entry is thread safe
#pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < n_docs; ++i) {
        const auto& [indices, values] = vectors_->get_vector_view_coded(i);
        size_t n_tokens = indices.size();
        size_t element_size = vectors_->get_element_size();
        for (size_t j = 0; j < n_tokens; ++j) {
            term_t term_id = indices[j];
            inverted_lists.add_entry(term_id, i,
                                     values.data() + j * element_size);
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

void SeismicIndex::search(idx_t n, const idx_t* indptr, const term_t* indices,
                          const float* values, int k, int cut,
                          float heap_factor, idx_t* labels) {
    throw_if_not_positive(n);
    throw_if_not_positive(k);
    throw_if_any_null(indptr, indices, values, labels);

    size_t indptr_size = n + 1;
    size_t nnz = indptr[n];  // Total non-zeros

    // Direct construction - single allocation + copy
    std::vector<idx_t> indptr_vec(indptr, indptr + indptr_size);
    std::vector<term_t> indices_vec(indices, indices + nnz);
    std::vector<float> values_vec(values, values + nnz);

    auto results =
        search(n, indptr_vec, indices_vec, values_vec, k, cut, heap_factor);

    idx_t* dest = labels;
    for (const auto& result : results) {
        dest = std::ranges::copy(result, dest).out;
    }
}

auto SeismicIndex::search(idx_t n, std::vector<idx_t>& indptr,
                          std::vector<term_t>& indices,
                          std::vector<float>& values, int k)
    -> std::vector<std::vector<idx_t>> {
    return search(n, indptr, indices, values, k, kDefaultCut,
                  kDefaultHeapFactor);
}

auto SeismicIndex::search(idx_t n, std::vector<idx_t>& indptr,
                          std::vector<term_t>& indices,
                          std::vector<float>& values, int k, int cut,
                          float heap_factor)
    -> std::vector<std::vector<idx_t>> {
    if (vectors_ == nullptr || n == 0) {
        return std::vector<std::vector<idx_t>>(n);
    }
    // Create query vectors from input
    SparseVectors query_vectors({.element_size = ElementSize::U32,
                                 .dimension = static_cast<size_t>(dimension_)});
    query_vectors.add_vectors(indptr, indices, values);
    std::vector<std::vector<idx_t>> results(n);
    double total_single_query_time_ms = 0.0;

    // For each query vector
#pragma omp parallel for num_threads(8)
    for (idx_t query_idx = 0; query_idx < n; ++query_idx) {
        const auto& dense = query_vectors.get_dense_vector_float(query_idx);
        const auto& cuts =
            top_k_tokens(query_vectors.get_vector_view(query_idx), cut);
        results[query_idx] =
            std::move(single_query(dense, cuts, k, heap_factor));
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
    std::unordered_set<idx_t> visited;
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
}  // namespace amaiss