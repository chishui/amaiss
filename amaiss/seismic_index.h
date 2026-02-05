#ifndef SEISMIC_INDEX_H
#define SEISMIC_INDEX_H
#include <array>
#include <vector>

#include "amaiss/cluster/inverted_list_clusters.h"
#include "amaiss/index.h"
#include "amaiss/io/io.h"
#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"

namespace amaiss {

struct SeismicSearchParameters : public SearchParameters {
    int cut = 10;
    float heap_factor = 1.0F;
    SeismicSearchParameters(int cut, float heap_factor)
        : cut(cut), heap_factor(heap_factor) {}
    SeismicSearchParameters() = default;
};

class SeismicIndex : public Index, public IndexIO {
    friend void write_index(Index* index, char* filename);
    friend Index* read_index(char* filename);

public:
    static constexpr std::array<char, 4> name = {'S', 'E', 'I', 'S'};

    explicit SeismicIndex(int dim = 0);
    SeismicIndex(int lambda, int beta, float alpha, int dim = 0);
    ~SeismicIndex() override = default;
    std::array<char, 4> id() const override { return name; }

    SeismicIndex(const SeismicIndex&) = delete;
    SeismicIndex& operator=(const SeismicIndex&) = delete;

    const SparseVectors* get_vectors() const override;
    void build() override;

    void add(idx_t n, const idx_t* indptr, const term_t* indices,
             const float* values) override;

protected:
    std::vector<InvertedListClusters> clustered_inverted_lists;

private:
    // override of IndexIO
    void write_index(IOWriter* io_writer) override;
    void read_index(IOReader* io_reader) override;

    auto search(idx_t n, const idx_t* indptr, const term_t* indices,
                const float* values, int k,
                const SearchParameters* search_parameters = nullptr)
        -> std::vector<std::vector<idx_t>> override;

    auto single_query(const std::vector<float>& dense,
                      const std::vector<term_t>& cuts, int k, float heap_factor)
        -> std::vector<idx_t>;

    std::unique_ptr<SparseVectors> vectors_;
    int lambda_;
    int beta_;
    float alpha_;
};
}  // namespace amaiss

#endif  // SEISMIC_INDEX_H