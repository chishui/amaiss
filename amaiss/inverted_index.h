#ifndef INVERTED_INDEX_H
#define INVERTED_INDEX_H

#include <array>
#include <memory>
#include <vector>

#include "amaiss/index.h"
#include "amaiss/invlists/inverted_lists.h"
#include "amaiss/io/io.h"
#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"

namespace amaiss {

class InvertedIndex : public Index, public IndexIO {
public:
    explicit InvertedIndex(int dim);

    InvertedIndex(const InvertedIndex&) = delete;
    InvertedIndex& operator=(const InvertedIndex&) = delete;
    void add(idx_t n, const idx_t* indptr, const term_t* indices,
             const float* values) override;
    void build() override;
    std::array<char, 4> id() const override { return name; }
    static constexpr std::array<char, 4> name = {'I', 'N', 'V', 'T'};

protected:
    auto search(idx_t n, const idx_t* indptr, const term_t* indices,
                const float* values, int k,
                SearchParameters* search_parameters = nullptr)
        -> pair_of_score_id_vectors_t override;

private:
    // IndexIO overrides
    void write_index(IOWriter* io_writer) override;
    void read_index(IOReader* io_reader) override;

    auto single_query(const term_t* indices, const float* values, int size,
                      int k) -> pair_of_score_id_vector_t;
    std::unique_ptr<ArrayInvertedLists> inverted_lists_;
    std::unique_ptr<SparseVectors> vectors_;
    // Per-term max posting value, computed at build() time.
    // max_term_scores_[term_id] = max value in that term's posting list.
    std::vector<float> max_term_scores_;
};

}  // namespace amaiss
#endif  // INVERTED_INDEX_H