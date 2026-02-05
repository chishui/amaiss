#include "amaiss/utils/vector_process.h"

#include <gtest/gtest.h>

#include <vector>

TEST(TopKTokens, basic) {
    std::vector<amaiss::term_t> indices = {0, 1, 2, 3, 4};
    std::vector<float> weights = {1.0F, 5.0F, 3.0F, 2.0F, 4.0F};

    auto result = amaiss::top_k_tokens(indices.data(), weights.data(), 5, 3);

    ASSERT_EQ(result.size(), 3);
    ASSERT_EQ(result[0], 1);  // weight 5.0 (highest)
    ASSERT_EQ(result[1], 4);  // weight 4.0
    ASSERT_EQ(result[2], 2);  // weight 3.0
}

TEST(TopKTokens, k_equals_size) {
    std::vector<amaiss::term_t> indices = {10, 20, 30};
    std::vector<float> weights = {3.0F, 1.0F, 2.0F};

    auto result = amaiss::top_k_tokens(indices.data(), weights.data(), 3, 3);

    ASSERT_EQ(result.size(), 3);
    ASSERT_EQ(result[0], 10);  // weight 3.0 (highest)
    ASSERT_EQ(result[1], 30);  // weight 2.0
    ASSERT_EQ(result[2], 20);  // weight 1.0
}

TEST(TopKTokens, k_greater_than_size) {
    std::vector<amaiss::term_t> indices = {5, 6};
    std::vector<float> weights = {2.0F, 1.0F};

    auto result = amaiss::top_k_tokens(indices.data(), weights.data(), 2, 5);

    ASSERT_EQ(result.size(), 5);
    // First 2 are actual values (descending), rest are default (0)
    ASSERT_EQ(result[0], 5);  // weight 2.0 (highest)
    ASSERT_EQ(result[1], 6);  // weight 1.0
    ASSERT_EQ(result[2], 0);
    ASSERT_EQ(result[3], 0);
    ASSERT_EQ(result[4], 0);
}

TEST(TopKTokens, single_element) {
    std::vector<amaiss::term_t> indices = {42};
    std::vector<float> weights = {7.0F};

    auto result = amaiss::top_k_tokens(indices.data(), weights.data(), 1, 1);

    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0], 42);
}

TEST(TopKTokens, duplicate_weights) {
    std::vector<amaiss::term_t> indices = {1, 2, 3, 4};
    std::vector<float> weights = {5.0F, 5.0F, 5.0F, 5.0F};

    auto result = amaiss::top_k_tokens(indices.data(), weights.data(), 4, 2);

    ASSERT_EQ(result.size(), 2);
}
