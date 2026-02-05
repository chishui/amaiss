#ifndef INDEX_H
#define INDEX_H

#include <array>
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
    virtual std::array<char, 4> id() const = 0;
    virtual void add(idx_t n, const idx_t* indptr, const term_t* indices,
                     const float* values) = 0;
    virtual void build();
    virtual void search(idx_t n, const idx_t* indptr, const term_t* indices,
                        const float* values, int k, idx_t* labels,
                        const SearchParameters* search_parameters =
                            nullptr);  // Pre-allocated: n * k
    virtual const SparseVectors* get_vectors() const = 0;

    int get_dimension() const { return dimension_; }
    size_t num_vectors() const {
        const auto* vectors = get_vectors();
        return vectors == nullptr ? 0 : vectors->num_vectors();
    }
    virtual void add_with_ids(idx_t n, const idx_t* indptr,
                              const term_t* indices, const float* values,
                              const idx_t* ids);
protected:
    virtual auto search(idx_t n, const idx_t* indptr, const term_t* indices,
                        const float* values, int k,
                        const SearchParameters* search_parameters = nullptr)
        -> std::vector<std::vector<idx_t>>;

    int dimension_;
};

}  // namespace amaiss

#endif  // INDEX_H