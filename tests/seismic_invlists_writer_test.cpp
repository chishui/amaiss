#include "amaiss/io/seismic_invlists_writer.h"

#include <gtest/gtest.h>

#include <vector>

#include "amaiss/cluster/inverted_list_clusters.h"
#include "amaiss/io/buffered_io.h"
#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"

namespace {

amaiss::SparseVectors create_float_vectors(
    const std::vector<std::vector<amaiss::term_t>>& indices_list,
    const std::vector<std::vector<float>>& values_list, size_t dimension = 10) {
    amaiss::SparseVectors vectors(
        {.element_size = amaiss::U32, .dimension = dimension});
    for (size_t i = 0; i < indices_list.size(); ++i) {
        const auto& indices = indices_list[i];
        const auto& values = values_list[i];
        vectors.add_vector(indices.data(), indices.size(),
                           reinterpret_cast<const uint8_t*>(values.data()),
                           values.size() * sizeof(float));
    }
    return vectors;
}

}  // namespace

TEST(SeismicInvertedListsWriter, serialize_deserialize_empty) {
    std::vector<amaiss::InvertedListClusters> original;
    amaiss::SeismicInvertedListsWriter writer(original);

    amaiss::BufferedIOWriter io_writer;
    writer.serialize(&io_writer);

    amaiss::BufferedIOReader io_reader(io_writer.data());
    std::vector<amaiss::InvertedListClusters> loaded;
    amaiss::SeismicInvertedListsWriter reader(loaded);
    reader.deserialize(&io_reader);

    auto result = reader.release();
    ASSERT_TRUE(result.empty());
}

TEST(SeismicInvertedListsWriter, serialize_deserialize_single_cluster) {
    std::vector<std::vector<amaiss::idx_t>> docs = {{0, 1}, {2}};
    amaiss::InvertedListClusters clusters(docs);

    std::vector<amaiss::InvertedListClusters> original;
    original.push_back(std::move(clusters));

    amaiss::SeismicInvertedListsWriter writer(original);

    amaiss::BufferedIOWriter io_writer;
    writer.serialize(&io_writer);

    amaiss::BufferedIOReader io_reader(io_writer.data());
    std::vector<amaiss::InvertedListClusters> loaded;
    amaiss::SeismicInvertedListsWriter reader(loaded);
    reader.deserialize(&io_reader);

    auto result = reader.release();
    ASSERT_EQ(result.size(), 1);

    auto doc_span0 = result[0].get_docs(0);
    ASSERT_EQ(doc_span0.size(), 2);
    ASSERT_EQ(doc_span0[0], 0);
    ASSERT_EQ(doc_span0[1], 1);

    auto doc_span1 = result[0].get_docs(1);
    ASSERT_EQ(doc_span1.size(), 1);
    ASSERT_EQ(doc_span1[0], 2);
}

TEST(SeismicInvertedListsWriter, serialize_deserialize_multiple_clusters) {
    std::vector<amaiss::InvertedListClusters> original;

    std::vector<std::vector<amaiss::idx_t>> docs1 = {{0, 1}};
    original.emplace_back(docs1);

    std::vector<std::vector<amaiss::idx_t>> docs2 = {{2, 3, 4}};
    original.emplace_back(docs2);

    std::vector<std::vector<amaiss::idx_t>> docs3 = {{5}};
    original.emplace_back(docs3);

    amaiss::SeismicInvertedListsWriter writer(original);

    amaiss::BufferedIOWriter io_writer;
    writer.serialize(&io_writer);

    amaiss::BufferedIOReader io_reader(io_writer.data());
    std::vector<amaiss::InvertedListClusters> loaded;
    amaiss::SeismicInvertedListsWriter reader(loaded);
    reader.deserialize(&io_reader);

    auto result = reader.release();
    ASSERT_EQ(result.size(), 3);

    ASSERT_EQ(result[0].get_docs(0).size(), 2);
    ASSERT_EQ(result[1].get_docs(0).size(), 3);
    ASSERT_EQ(result[2].get_docs(0).size(), 1);
}

TEST(SeismicInvertedListsWriter, serialize_deserialize_with_summaries) {
    std::vector<std::vector<amaiss::idx_t>> docs = {{0, 1}, {2}};
    amaiss::InvertedListClusters clusters(docs);

    auto vectors = create_float_vectors(
        {{0, 1}, {0, 2}, {1, 2}}, {{1.0F, 2.0F}, {1.5F, 1.0F}, {3.0F, 2.0F}});
    clusters.summarize(&vectors, 1.0F);

    std::vector<amaiss::InvertedListClusters> original;
    original.push_back(std::move(clusters));

    amaiss::SeismicInvertedListsWriter writer(original);

    amaiss::BufferedIOWriter io_writer;
    writer.serialize(&io_writer);

    amaiss::BufferedIOReader io_reader(io_writer.data());
    std::vector<amaiss::InvertedListClusters> loaded;
    amaiss::SeismicInvertedListsWriter reader(loaded);
    reader.deserialize(&io_reader);

    auto result = reader.release();
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0].cluster_size(), 2);
}

TEST(SeismicInvertedListsWriter, release_moves_data) {
    std::vector<std::vector<amaiss::idx_t>> docs = {{0, 1}};
    amaiss::InvertedListClusters clusters(docs);

    std::vector<amaiss::InvertedListClusters> original;
    original.push_back(std::move(clusters));

    amaiss::SeismicInvertedListsWriter writer(original);

    auto released = writer.release();
    ASSERT_EQ(released.size(), 1);
}
