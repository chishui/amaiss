#ifndef SEISMIC_INVLISTS_WRITER_H
#define SEISMIC_INVLISTS_WRITER_H
#include <vector>

#include "amaiss/cluster/inverted_list_clusters.h"
#include "amaiss/io/index_io.h"

namespace amaiss {
class SeismicInvertedListsWriter : public Serializable {
public:
    SeismicInvertedListsWriter(
        const std::vector<InvertedListClusters>& clustered_inverted_lists)
        : clustered_inverted_lists_(clustered_inverted_lists) {}

    void serialize(IOWriter* writer) const override {
        size_t size = clustered_inverted_lists_.size();
        writer->write(&size, sizeof(size), 1);
        for (const auto& clusters : clustered_inverted_lists_) {
            clusters.serialize(writer);
        }
    }
    void deserialize(IOReader* reader) override {
        size_t size = 0;
        reader->read(&size, sizeof(size), 1);
        clustered_inverted_lists_.resize(size);
        for (auto& clusters : clustered_inverted_lists_) {
            clusters.deserialize(reader);
        }
    }

    std::vector<InvertedListClusters>&& release() {
        return std::move(clustered_inverted_lists_);
    }

private:
    std::vector<InvertedListClusters> clustered_inverted_lists_;
};
}  // namespace amaiss

#endif  // SEISMIC_INVLISTS_WRITER_H