#ifndef SEISMIC_INDEX_H
#define SEISMIC_INDEX_H
#include <vector>

#include "amaiss/cluster/inverted_list_clusters.h"
#include "amaiss/index.h"
#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"

namespace amaiss {
class SeismicIndex : public Index {
    static constexpr int kDefaultCut = 3;
    static constexpr float kDefaultHeapFactor = 1.0F;

public:
    explicit SeismicIndex(int dim = 0);
    SeismicIndex(int lambda, int beta, float alpha, int dim = 0);
    ~SeismicIndex() override = default;

    SeismicIndex(const SeismicIndex&) = delete;
    SeismicIndex& operator=(const SeismicIndex&) = delete;

    const SparseVectors* get_vectors() const override;
    void build();

    void search(idx_t n, const idx_t* indptr, const term_t* indices,
                const float* values, int k, int cut, float heap_factor,
                idx_t* labels);

private:
    void add(idx_t n, std::vector<idx_t>& indptr, std::vector<term_t>& indices,
             std::vector<float>& values) override;

    auto search(idx_t n, std::vector<idx_t>& indptr,
                std::vector<term_t>& indices, std::vector<float>& values, int k)
        -> std::vector<std::vector<idx_t>> override;

    auto search(idx_t n, std::vector<idx_t>& indptr,
                std::vector<term_t>& indices, std::vector<float>& values, int k,
                int cut, float heap_factor) -> std::vector<std::vector<idx_t>>;

    auto single_query(const std::vector<float>& dense,
                      const std::vector<term_t>& cuts, int k, float heap_factor)
        -> std::vector<idx_t>;

    std::unique_ptr<SparseVectors> vectors_;
    int lambda_;
    int beta_;
    float alpha_;
    std::vector<InvertedListClusters> clustered_inverted_lists;
};
}  // namespace amaiss

#endif  // SEISMIC_INDEX_H