#ifndef ID_MAP_INDEX_H
#define ID_MAP_INDEX_H
#include <array>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "amaiss/id_selector.h"
#include "amaiss/index.h"
#include "amaiss/io/io.h"
#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"

namespace amaiss {

class IDSelectorWithIDMap : public IDSelector {
public:
    IDSelectorWithIDMap(const IDSelector* id_selector,
                        const std::vector<idx_t>& id_maps)
        : delegate_(id_selector), id_map_(id_maps) {}

    bool is_member(idx_t id) const override {
        return delegate_->is_member(id_map_[id]);
    }

private:
    const IDSelector* delegate_;
    const std::vector<idx_t>& id_map_;
};

class IDMapIndex : public Index, public IndexIO {
public:
    IDMapIndex() = default;
    static constexpr std::array<char, 4> name = {'I', 'D', 'M', 'P'};
    explicit IDMapIndex(Index*);
    std::array<char, 4> id() const override { return name; }

    void add(idx_t n, const idx_t* indptr, const term_t* indices,
             const float* values) override;
    void build() override;
    void search(idx_t n, const idx_t* indptr, const term_t* indices,
                const float* values, int k, float* distances, idx_t* labels,
                SearchParameters* search_parameters = nullptr) override;
    const SparseVectors* get_vectors() const override;

    void add_with_ids(idx_t n, const idx_t* indptr, const term_t* indices,
                      const float* values, const idx_t* ids) override;
    void write_index(IOWriter* io_writer) override;
    void read_index(IOReader* io_reader) override;

private:
    Index* delegate_ = nullptr;
    std::vector<idx_t> internal_to_external_;
    absl::flat_hash_map<idx_t, idx_t> external_to_internal_;
};
}  // namespace amaiss

#endif  // ID_MAP_INDEX_H