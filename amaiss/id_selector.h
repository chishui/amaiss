#ifndef ID_SELECTOR_H
#define ID_SELECTOR_H

#include <unordered_set>

#include "amaiss/types.h"

namespace amaiss {
class IDSelector {
public:
    virtual ~IDSelector() = default;
    virtual bool is_member(idx_t id) const = 0;
    bool operator()(idx_t id) const { return is_member(id); }
};

class SetIDSelector : public IDSelector {
public:
    explicit SetIDSelector(size_t n, const idx_t* indices) {
        ids_ = std::unordered_set<idx_t>(indices, indices + n);
    }
    bool is_member(idx_t id) const override { return ids_.contains(id); }

private:
    std::unordered_set<idx_t> ids_;
};

class NotIDSelector : public IDSelector {
public:
    explicit NotIDSelector(IDSelector* selector) : delegate_(selector) {}
    bool is_member(idx_t id) const override {
        return !delegate_->is_member(id);
    }

private:
    IDSelector* delegate_;
};
}  // namespace amaiss

#endif  // ID_SELECTOR_H