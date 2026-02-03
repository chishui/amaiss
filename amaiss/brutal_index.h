#ifndef BRUTAL_INDEX_H
#define BRUTAL_INDEX_H

#include <memory>
#include <vector>

#include "amaiss/index.h"
#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"

namespace amaiss {

class BrutalIndex : public Index {
public:
    explicit BrutalIndex(int dim = 0);

    BrutalIndex(const BrutalIndex&) = delete;
    BrutalIndex& operator=(const BrutalIndex&) = delete;
    void add(idx_t n, const idx_t* indptr, const term_t* indices,
             const float* values) override;

protected:
    auto search(idx_t n, const idx_t* indptr, const term_t* indices,
                const float* values, int k,
                const SearchParameters* search_parameters = nullptr)
        -> std::vector<std::vector<idx_t>> override;
    const SparseVectors* get_vectors() const override;

private:
    auto single_query(const std::vector<float>& dense, int k)
        -> std::vector<idx_t>;
    std::unique_ptr<SparseVectors> vectors_;
};

}  // namespace amaiss
#endif  // BRUTAL_INDEX_H