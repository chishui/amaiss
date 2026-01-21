#ifndef INDEX_H
#define INDEX_H

#include <vector>

#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"

namespace amaiss {

class Index {
public:
    explicit Index(int dim = 0);
    virtual ~Index() = default;
    virtual void add(idx_t n, const idx_t* indptr, const term_t* indices,
                     const float* values);
    virtual void search(idx_t n, const idx_t* indptr, const term_t* indices,
                        const float* values, int k,
                        idx_t* labels);  // Pre-allocated: n * k
    virtual const SparseVectors* get_vectors() const { return nullptr; }

    int get_dimension() const { return dimension_; }

protected:
    virtual void add(idx_t n, std::vector<idx_t>& indptr,
                     std::vector<term_t>& indices,
                     std::vector<float>& values) = 0;

    virtual auto search(idx_t n, std::vector<idx_t>& indptr,
                        std::vector<term_t>& indices,
                        std::vector<float>& values, int k)
        -> std::vector<std::vector<idx_t>> = 0;

    int dimension_;
};

}  // namespace amaiss

#endif  // INDEX_H