#ifndef SEISMIC_SCALAR_QUANTIZED_INDEX_H
#define SEISMIC_SCALAR_QUANTIZED_INDEX_H

#include <memory>
#include <vector>

#include "amaiss/cluster/inverted_list_clusters.h"
#include "amaiss/index.h"
#include "amaiss/seismic_index.h"
#include "amaiss/utils/scalar_quantizer.h"

namespace amaiss {

struct SeismicSQSearchParameters : public SeismicSearchParameters {
    float vmin;
    float vmax;
    SeismicSQSearchParameters(float vmin, float vmax, int cut,
                              float heap_factor)
        : SeismicSearchParameters(cut, heap_factor), vmax(vmax), vmin(vmin) {}
};

class SeismicScalarQuantizedIndex : public Index {
public:
    SeismicScalarQuantizedIndex(QuantizerType quantizer_type, float vmin,
                                float vmax, int lambda, int beta, float alpha,
                                int dim);
    ~SeismicScalarQuantizedIndex() override = default;

    SeismicScalarQuantizedIndex(const SeismicScalarQuantizedIndex&) = delete;
    SeismicScalarQuantizedIndex& operator=(const SeismicScalarQuantizedIndex&) =
        delete;

    void add(idx_t n, const idx_t* indptr, const term_t* indices,
             const float* values) override;
    void build() override;
    const SparseVectors* get_vectors() const override { return vectors_.get(); }

private:
    auto search(idx_t n, const idx_t* indptr, const term_t* indices,
                const float* values, int k,
                const SearchParameters* search_parameters = nullptr)
        -> std::vector<std::vector<idx_t>> override;
    auto encode(const float* values, size_t nnz,
                const SearchParameters* search_parameters)
        -> std::vector<uint8_t>;
    auto single_query(const std::vector<uint8_t>& dense,
                      const std::vector<term_t>& cuts, int k, float heap_factor)
        -> std::vector<idx_t>;
    ScalarQuantizer sq;
    std::unique_ptr<SparseVectors> vectors_;
    int lambda_;
    int beta_;
    float alpha_;
    std::vector<InvertedListClusters> clustered_inverted_lists;
};
}  // namespace amaiss

#endif  // SEISMIC_SCALAR_QUANTIZED_INDEX_H