#include "amaiss/seismic_scalar_quantized_index.h"

#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <typeinfo>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "amaiss/cluster/inverted_list_clusters.h"
#include "amaiss/cluster/random_kmeans.h"
#include "amaiss/index.h"
#include "amaiss/invlists/inverted_lists.h"
#include "amaiss/io/io.h"
#include "amaiss/io/seismic_invlists_writer.h"
#include "amaiss/types.h"
#include "amaiss/utils/checks.h"
#include "amaiss/utils/distance_simd.h"
#include "amaiss/utils/prefetch.h"
#include "amaiss/utils/scalar_quantizer.h"
#include "amaiss/utils/vector_process.h"

namespace amaiss {

static void query_single_inverted_list(
    const SparseVectors* vectors, const InvertedListClusters& cluster_invlist,
    const std::vector<uint8_t>& dense, TopKHolder<idx_t>& heap,
    absl::flat_hash_set<idx_t>& visited, float heap_factor, bool first_list) {
    // Skip empty clusters
    size_t csize = cluster_invlist.cluster_size();
    if (csize == 0) {
        return;
    }
    const auto element_size = vectors->get_element_size();
    const auto& summaries = cluster_invlist.summaries();
    // compute dp with all summaries
    std::vector<float> summary_scores;
    if (element_size == U16) {
        summary_scores = dot_product_uint16_vectors_dense(
            &summaries, reinterpret_cast<const uint16_t*>(dense.data()));
    } else {
        summary_scores =
            dot_product_uint8_vectors_dense(&summaries, dense.data());
    }
    size_t num_vectors = vectors->num_vectors();

    std::vector<size_t> cluster_order(summary_scores.size());
    std::iota(cluster_order.begin(), cluster_order.end(), 0);
    if (first_list) {
        std::ranges::sort(cluster_order, [&](size_t a, size_t b) {
            return summary_scores[a] > summary_scores[b];
        });
    }

    const auto* indptr = vectors->indptr_data();
    const auto* indices = vectors->indices_data();
    const auto* values = vectors->values_data();

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
            float score = 0;
            if (element_size == U16) {
                // start is element index, need to convert to byte offset for
                // uint16_t access
                const auto* int16_values = reinterpret_cast<const uint16_t*>(
                    values + start * sizeof(uint16_t));
                const auto* int16_dense =
                    reinterpret_cast<const uint16_t*>(dense.data());
                score = dot_product_uint16_dense(indices + start, int16_values,
                                                 len, int16_dense);
            } else {
                score = dot_product_uint8_dense(indices + start, values + start,
                                                len, dense.data());
            }
            heap.add(score, doc_id);
        }
    }
}
SeismicScalarQuantizedIndex::SeismicScalarQuantizedIndex(int dim)
    : Index(dim) {}

SeismicScalarQuantizedIndex::SeismicScalarQuantizedIndex(
    QuantizerType quantizer_type, float vmin, float vmax, int lambda, int beta,
    float alpha, int dim)
    : Index(dim),
      sq_(quantizer_type, vmin, vmax),
      lambda_(lambda),
      beta_(beta),
      alpha_(alpha) {}

void SeismicScalarQuantizedIndex::add(idx_t n, const idx_t* indptr,
                                      const term_t* indices,
                                      const float* values) {
    throw_if_not_positive(n);
    throw_if_any_null(indptr, indices, values);

    size_t indptr_size = n + 1;
    size_t nnz = indptr[n];  // Total non-zeros
    const size_t element_size = sq_.bytes_per_value();
    if (vectors_ == nullptr) {
        vectors_ = std::unique_ptr<SparseVectors>(
            new SparseVectors({.element_size = element_size,
                               .dimension = static_cast<size_t>(dimension_)}));
    }
    std::vector<uint8_t> codes(nnz * element_size);
    sq_.encode(values, codes.data(), nnz);
    vectors_->add_vectors(indptr, indptr_size, indices, nnz, codes.data(),
                          nnz * element_size);
}

// encode based on search_parameters type, if it's SeismicSearchParameters,
// use Index's quantizer, if it's SeismicSQSearchParameters, construct
// quantizer using SeismicSQSearchParameters's parameters
std::vector<uint8_t> SeismicScalarQuantizedIndex::encode(
    const float* values, size_t nnz,
    const SearchParameters* search_parameters) {
    const size_t element_size = sq_.bytes_per_value();
    std::vector<uint8_t> codes(nnz * element_size);
    if (typeid(*search_parameters) == typeid(SeismicSearchParameters)) {
        sq_.encode(values, codes.data(), nnz);
    } else if (typeid(*search_parameters) ==
               typeid(SeismicSQSearchParameters)) {
        const auto* seismic_sq_search_parameters =
            static_cast<const SeismicSQSearchParameters*>(search_parameters);
        ScalarQuantizer search_sq(sq_.get_quantizer_type(),
                                  seismic_sq_search_parameters->vmin,
                                  seismic_sq_search_parameters->vmax);
        search_sq.encode(values, codes.data(), nnz);
    } else {
        throw std::runtime_error("Unsupported search parameters type!");
    }
    return codes;
}

void SeismicScalarQuantizedIndex::build() {
    // build inverted index
    std::unique_ptr<ArrayInvertedLists> inverted_lists =
        ArrayInvertedLists::build_inverted_lists(
            get_dimension(), sq_.bytes_per_value(), vectors_.get());

    // TODO: generate lambda and beta
    size_t num_inverted_lists = inverted_lists->size();
    clustered_inverted_lists.clear();
    clustered_inverted_lists.resize(num_inverted_lists);
#pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < num_inverted_lists; ++idx) {
        auto& invlist = (*inverted_lists)[idx];
        const auto& doc_ids = invlist.prune_and_keep_doc_ids(lambda_);
        InvertedListClusters inverted_list_clusters(
            RandomKMeans::train(vectors_.get(), doc_ids, beta_));
        inverted_list_clusters.summarize(vectors_.get(), alpha_);
        clustered_inverted_lists[idx] = std::move(inverted_list_clusters);
        invlist.clear();
    }
}

auto SeismicScalarQuantizedIndex::search(
    idx_t n, const idx_t* indptr, const term_t* indices, const float* values,
    int k, const SearchParameters* search_parameters)
    -> std::vector<std::vector<idx_t>> {
    throw_if_null(search_parameters, "search parameters cannot be null!");
    if (vectors_ == nullptr || n == 0) {
        return std::vector<std::vector<idx_t>>(n);
    }
    size_t indptr_size = n + 1;
    size_t nnz = indptr[n];  // Total non-zeros

    // construct query vector
    const size_t element_size = sq_.bytes_per_value();
    SparseVectors query_vectors({.element_size = element_size,
                                 .dimension = static_cast<size_t>(dimension_)});
    std::vector<uint8_t> codes = encode(values, nnz, search_parameters);
    query_vectors.add_vectors(indptr, indptr_size, indices, nnz, codes.data(),
                              nnz * element_size);

    std::vector<std::vector<idx_t>> results(n);

    // query
    const auto* parameters =
        dynamic_cast<const SeismicSearchParameters*>(search_parameters);
    const auto* query_indptr = query_vectors.indptr_data();
    const auto* query_indices = query_vectors.indices_data();
    const auto* query_values = query_vectors.values_data();

    int num_threads = 8;
    int chunk_size = std::max(1, static_cast<int>(n) / (num_threads * 4));
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (idx_t query_idx = 0; query_idx < n; ++query_idx) {
        const auto& dense = query_vectors.get_dense_vector(query_idx);
        const idx_t start = query_indptr[query_idx];
        const size_t len = query_indptr[query_idx + 1] - start;
        std::vector<term_t> cuts;
        if (element_size == U16) {
            // start is element index, need byte offset for uint16_t access
            cuts = top_k_tokens<uint16_t>(
                query_indices + start,
                reinterpret_cast<const uint16_t*>(query_values +
                                                  start * sizeof(uint16_t)),
                len, parameters->cut);
        } else {
            cuts = top_k_tokens<uint8_t>(query_indices + start,
                                         query_values + start, len,
                                         parameters->cut);
        }

        results[query_idx] =
            std::move(single_query(dense, cuts, k, parameters->heap_factor));
    }

    return results;
}

auto SeismicScalarQuantizedIndex::single_query(
    const std::vector<uint8_t>& dense, const std::vector<term_t>& cuts, int k,
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

    return holder.top_k_descending_with_padding(INVALID_IDX);
}

void SeismicScalarQuantizedIndex::write_index(IOWriter* io_writer) {
    write_header(io_writer);
    // write vectors
    if (vectors_ == nullptr) {
        empty_sparse_vectors.serialize(io_writer);
    } else {
        vectors_->serialize(io_writer);
    }
    SeismicInvertedListsWriter inv_list_writer(clustered_inverted_lists);
    inv_list_writer.serialize(io_writer);
}

void SeismicScalarQuantizedIndex::read_index(IOReader* io_reader) {
    read_header(io_reader);
    SparseVectors tmp_vectors;
    tmp_vectors.deserialize(io_reader);
    if (tmp_vectors.num_vectors() > 0) {
        vectors_ = std::make_unique<SparseVectors>(std::move(tmp_vectors));
    }
    SeismicInvertedListsWriter inv_list_writer({});
    inv_list_writer.deserialize(io_reader);
    clustered_inverted_lists = std::move(inv_list_writer.release());
}

void SeismicScalarQuantizedIndex::write_header(IOWriter* io_writer) {
    auto sq_type = sq_.get_quantizer_type();
    io_writer->write(&sq_type, sizeof(QuantizerType), 1);
    auto vmin = sq_.get_min();
    io_writer->write(&vmin, sizeof(float), 1);
    auto vmax = sq_.get_max();
    io_writer->write(&vmax, sizeof(float), 1);
}

void SeismicScalarQuantizedIndex::read_header(IOReader* io_reader) {
    QuantizerType sq_type = QuantizerType::QT_8bit;
    float vmin = 0.0F;
    float vmax = 1.0F;
    io_reader->read(&sq_type, sizeof(QuantizerType), 1);
    io_reader->read(&vmin, sizeof(float), 1);
    io_reader->read(&vmax, sizeof(float), 1);
    sq_ = ScalarQuantizer(sq_type, vmin, vmax);
}
}  // namespace amaiss
