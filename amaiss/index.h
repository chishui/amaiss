#ifndef INDEX_H
#define INDEX_H

#include <vector>

#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"

namespace amaiss {

struct SearchParameters {
    virtual ~SearchParameters() = default;
};

class Index {
public:
    explicit Index(int dim = 0);
    virtual ~Index() = default;
    virtual void add(idx_t n, const idx_t* indptr, const term_t* indices,
                     const float* values) = 0;
    virtual void build();
    virtual void search(idx_t n, const idx_t* indptr, const term_t* indices,
                        const float* values, int k, idx_t* labels,
                        const SearchParameters* search_parameters =
                            nullptr);  // Pre-allocated: n * k
    virtual const SparseVectors* get_vectors() const = 0;

    int get_dimension() const { return dimension_; }

protected:
    virtual auto search(idx_t n, const idx_t* indptr, const term_t* indices,
                        const float* values, int k,
                        const SearchParameters* search_parameters = nullptr)
        -> std::vector<std::vector<idx_t>>;

    int dimension_;
};

}  // namespace amaiss

#endif  // INDEX_H