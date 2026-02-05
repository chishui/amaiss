#include "amaiss/id_map_index.h"

#include <gtest/gtest.h>

#include <vector>

#include "amaiss/seismic_index.h"
#include "amaiss/types.h"

namespace {

class IDMapIndexTest : public ::testing::Test {
protected:
    void SetUp() override {
        seismic_ = new amaiss::SeismicIndex(10, 2, 0.5F, 100);
        idmap_ = new amaiss::IDMapIndex(seismic_);
    }

    void TearDown() override { delete idmap_; }

    amaiss::SeismicIndex* seismic_;
    amaiss::IDMapIndex* idmap_;
};

}  // namespace

TEST_F(IDMapIndexTest, id) {
    EXPECT_EQ(idmap_->id(), amaiss::IDMapIndex::name);
}

TEST_F(IDMapIndexTest, get_vectors_empty) {
    EXPECT_EQ(idmap_->get_vectors(), nullptr);
}

TEST_F(IDMapIndexTest, num_vectors_empty) {
    EXPECT_EQ(idmap_->num_vectors(), 0);
}

TEST_F(IDMapIndexTest, add_with_ids) {
    std::vector<amaiss::idx_t> indptr = {0, 2, 4};
    std::vector<amaiss::term_t> indices = {0, 1, 2, 3};
    std::vector<float> values = {1.0F, 0.5F, 0.8F, 0.3F};
    std::vector<amaiss::idx_t> ids = {100, 200};

    idmap_->add_with_ids(2, indptr.data(), indices.data(), values.data(),
                         ids.data());

    EXPECT_EQ(idmap_->num_vectors(), 2);
}

TEST_F(IDMapIndexTest, add_with_ids_multiple_batches) {
    std::vector<amaiss::idx_t> indptr1 = {0, 2};
    std::vector<amaiss::term_t> indices1 = {0, 1};
    std::vector<float> values1 = {1.0F, 0.5F};
    std::vector<amaiss::idx_t> ids1 = {100};

    idmap_->add_with_ids(1, indptr1.data(), indices1.data(), values1.data(),
                         ids1.data());
    EXPECT_EQ(idmap_->num_vectors(), 1);

    std::vector<amaiss::idx_t> indptr2 = {0, 2};
    std::vector<amaiss::term_t> indices2 = {2, 3};
    std::vector<float> values2 = {0.8F, 0.3F};
    std::vector<amaiss::idx_t> ids2 = {200};

    idmap_->add_with_ids(1, indptr2.data(), indices2.data(), values2.data(),
                         ids2.data());
    EXPECT_EQ(idmap_->num_vectors(), 2);
}

TEST_F(IDMapIndexTest, search_returns_external_ids) {
    // Add vectors with custom external IDs
    std::vector<amaiss::idx_t> indptr = {0, 2, 4, 6};
    std::vector<amaiss::term_t> indices = {0, 1, 0, 1, 0, 1};
    std::vector<float> values = {1.0F, 0.5F, 0.3F, 0.2F, 0.8F, 0.4F};
    std::vector<amaiss::idx_t> ids = {1000, 2000, 3000};

    idmap_->add_with_ids(3, indptr.data(), indices.data(), values.data(),
                         ids.data());
    idmap_->build();

    // Query
    std::vector<amaiss::idx_t> query_indptr = {0, 2};
    std::vector<amaiss::term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 1.0F};
    std::vector<amaiss::idx_t> labels(3, -1);

    idmap_->search(1, query_indptr.data(), query_indices.data(),
                   query_values.data(), 3, labels.data(), nullptr);

    // Results should be external IDs (1000, 2000, 3000), not internal (0, 1, 2)
    for (const auto& label : labels) {
        EXPECT_TRUE(label == 1000 || label == 2000 || label == 3000 ||
                    label == -1);
    }
}

TEST_F(IDMapIndexTest, search_preserves_negative_ids) {
    // Add one vector
    std::vector<amaiss::idx_t> indptr = {0, 2};
    std::vector<amaiss::term_t> indices = {0, 1};
    std::vector<float> values = {1.0F, 0.5F};
    std::vector<amaiss::idx_t> ids = {1000};

    idmap_->add_with_ids(1, indptr.data(), indices.data(), values.data(),
                         ids.data());
    idmap_->build();

    // Query for k=3 but only 1 result exists
    std::vector<amaiss::idx_t> query_indptr = {0, 2};
    std::vector<amaiss::term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 1.0F};
    std::vector<amaiss::idx_t> labels(3, -1);

    idmap_->search(1, query_indptr.data(), query_indices.data(),
                   query_values.data(), 3, labels.data(), nullptr);

    // First result should be external ID, rest should be -1 (padding)
    EXPECT_EQ(labels[0], 1000);
    EXPECT_EQ(labels[1], -1);
    EXPECT_EQ(labels[2], -1);
}

TEST_F(IDMapIndexTest, get_vectors_after_add) {
    std::vector<amaiss::idx_t> indptr = {0, 2};
    std::vector<amaiss::term_t> indices = {0, 1};
    std::vector<float> values = {1.0F, 0.5F};
    std::vector<amaiss::idx_t> ids = {100};

    idmap_->add_with_ids(1, indptr.data(), indices.data(), values.data(),
                         ids.data());

    EXPECT_NE(idmap_->get_vectors(), nullptr);
    EXPECT_EQ(idmap_->get_vectors()->num_vectors(), 1);
}

TEST(IDMapIndex, default_constructor) {
    amaiss::IDMapIndex idmap;
    EXPECT_EQ(idmap.get_vectors(), nullptr);
    EXPECT_EQ(idmap.num_vectors(), 0);
}
