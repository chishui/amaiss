#include "amaiss/invlists/inverted_lists.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"

// InvertedList tests
TEST(InvertedList, constructor) {
    amaiss::InvertedList list(amaiss::U32);
    ASSERT_TRUE(list.get_doc_ids().empty());
    ASSERT_TRUE(list.get_codes().empty());
}

TEST(InvertedList, add_entries_empty) {
    amaiss::InvertedList list(amaiss::U32);
    list.add_entries(0, nullptr, nullptr);
    ASSERT_TRUE(list.get_doc_ids().empty());
    ASSERT_TRUE(list.get_codes().empty());
}

TEST(InvertedList, add_entries_single_float) {
    amaiss::InvertedList list(amaiss::U32);
    amaiss::idx_t doc_id = 42;
    float value = 1.5F;
    list.add_entries(1, &doc_id, reinterpret_cast<const uint8_t*>(&value));

    ASSERT_EQ(list.get_doc_ids().size(), 1);
    ASSERT_EQ(list.get_doc_ids()[0], 42);
    ASSERT_EQ(list.get_codes().size(), sizeof(float));

    float stored = *reinterpret_cast<const float*>(list.get_codes().data());
    ASSERT_FLOAT_EQ(stored, 1.5F);
}

TEST(InvertedList, add_entries_multiple_float) {
    amaiss::InvertedList list(amaiss::U32);
    std::vector<amaiss::idx_t> doc_ids = {1, 2, 3};
    std::vector<float> values = {1.0F, 2.0F, 3.0F};

    list.add_entries(3, doc_ids.data(),
                     reinterpret_cast<const uint8_t*>(values.data()));

    ASSERT_EQ(list.get_doc_ids().size(), 3);
    ASSERT_EQ(list.get_doc_ids()[0], 1);
    ASSERT_EQ(list.get_doc_ids()[1], 2);
    ASSERT_EQ(list.get_doc_ids()[2], 3);

    ASSERT_EQ(list.get_codes().size(), 3 * sizeof(float));
    const auto* stored =
        reinterpret_cast<const float*>(list.get_codes().data());
    ASSERT_FLOAT_EQ(stored[0], 1.0F);
    ASSERT_FLOAT_EQ(stored[1], 2.0F);
    ASSERT_FLOAT_EQ(stored[2], 3.0F);
}

TEST(InvertedList, add_entries_uint8) {
    amaiss::InvertedList list(amaiss::U8);
    std::vector<amaiss::idx_t> doc_ids = {10, 20};
    std::vector<uint8_t> values = {100, 200};

    list.add_entries(2, doc_ids.data(), values.data());

    ASSERT_EQ(list.get_doc_ids().size(), 2);
    ASSERT_EQ(list.get_codes().size(), 2);
    ASSERT_EQ(list.get_codes()[0], 100);
    ASSERT_EQ(list.get_codes()[1], 200);
}

TEST(InvertedList, add_entries_uint16) {
    amaiss::InvertedList list(amaiss::U16);
    std::vector<amaiss::idx_t> doc_ids = {5, 6};
    std::vector<uint16_t> values = {1000, 2000};

    list.add_entries(2, doc_ids.data(),
                     reinterpret_cast<const uint8_t*>(values.data()));

    ASSERT_EQ(list.get_doc_ids().size(), 2);
    ASSERT_EQ(list.get_codes().size(), 2 * sizeof(uint16_t));

    const auto* stored =
        reinterpret_cast<const uint16_t*>(list.get_codes().data());
    ASSERT_EQ(stored[0], 1000);
    ASSERT_EQ(stored[1], 2000);
}

TEST(InvertedList, add_entries_accumulates) {
    amaiss::InvertedList list(amaiss::U32);

    amaiss::idx_t doc1 = 1;
    float val1 = 1.0F;
    list.add_entries(1, &doc1, reinterpret_cast<const uint8_t*>(&val1));

    amaiss::idx_t doc2 = 2;
    float val2 = 2.0F;
    list.add_entries(1, &doc2, reinterpret_cast<const uint8_t*>(&val2));

    ASSERT_EQ(list.get_doc_ids().size(), 2);
    ASSERT_EQ(list.get_doc_ids()[0], 1);
    ASSERT_EQ(list.get_doc_ids()[1], 2);
}

TEST(InvertedList, clear) {
    amaiss::InvertedList list(amaiss::U32);
    std::vector<amaiss::idx_t> doc_ids = {1, 2, 3};
    std::vector<float> values = {1.0F, 2.0F, 3.0F};
    list.add_entries(3, doc_ids.data(),
                     reinterpret_cast<const uint8_t*>(values.data()));

    list.clear();

    ASSERT_TRUE(list.get_doc_ids().empty());
    ASSERT_TRUE(list.get_codes().empty());
}

TEST(InvertedList, prune_and_keep_doc_ids_empty_list) {
    amaiss::InvertedList list(amaiss::U32);
    auto result = list.prune_and_keep_doc_ids(5);
    ASSERT_TRUE(result.empty());
}

TEST(InvertedList, prune_and_keep_doc_ids_lambda_zero) {
    amaiss::InvertedList list(amaiss::U32);
    std::vector<amaiss::idx_t> doc_ids = {1, 2, 3};
    std::vector<float> values = {1.0F, 2.0F, 3.0F};
    list.add_entries(3, doc_ids.data(),
                     reinterpret_cast<const uint8_t*>(values.data()));

    auto result = list.prune_and_keep_doc_ids(0);
    ASSERT_EQ(result.size(), 3);  // Returns all when lambda <= 0
}

TEST(InvertedList, prune_and_keep_doc_ids_lambda_exceeds_size) {
    amaiss::InvertedList list(amaiss::U32);
    std::vector<amaiss::idx_t> doc_ids = {1, 2};
    std::vector<float> values = {1.0F, 2.0F};
    list.add_entries(2, doc_ids.data(),
                     reinterpret_cast<const uint8_t*>(values.data()));

    auto result = list.prune_and_keep_doc_ids(10);
    ASSERT_EQ(result.size(), 2);  // Returns all when lambda >= n_docs
}

TEST(InvertedList, prune_and_keep_doc_ids_keeps_top_values_float) {
    amaiss::InvertedList list(amaiss::U32);
    // doc_ids: 10, 20, 30, 40 with values: 1.0, 4.0, 2.0, 3.0
    // Top 2 by value: doc 20 (4.0), doc 40 (3.0)
    std::vector<amaiss::idx_t> doc_ids = {10, 20, 30, 40};
    std::vector<float> values = {1.0F, 4.0F, 2.0F, 3.0F};
    list.add_entries(4, doc_ids.data(),
                     reinterpret_cast<const uint8_t*>(values.data()));

    auto result = list.prune_and_keep_doc_ids(2);

    ASSERT_EQ(result.size(), 2);
    // Should contain doc 20 and doc 40 (highest values)
    std::ranges::sort(result);
    ASSERT_EQ(result[0], 20);
    ASSERT_EQ(result[1], 40);
}

TEST(InvertedList, prune_and_keep_doc_ids_keeps_top_values_uint8) {
    amaiss::InvertedList list(amaiss::U8);
    // doc_ids: 1, 2, 3 with values: 50, 200, 100
    // Top 2 by value: doc 2 (200), doc 3 (100)
    std::vector<amaiss::idx_t> doc_ids = {1, 2, 3};
    std::vector<uint8_t> values = {50, 200, 100};
    list.add_entries(3, doc_ids.data(), values.data());

    auto result = list.prune_and_keep_doc_ids(2);

    ASSERT_EQ(result.size(), 2);
    std::ranges::sort(result);
    ASSERT_EQ(result[0], 2);
    ASSERT_EQ(result[1], 3);
}

TEST(InvertedList, prune_and_keep_doc_ids_keeps_top_values_uint16) {
    amaiss::InvertedList list(amaiss::U16);
    // doc_ids: 1, 2, 3 with values: 500, 2000, 1000
    // Top 2 by value: doc 2 (2000), doc 3 (1000)
    std::vector<amaiss::idx_t> doc_ids = {1, 2, 3};
    std::vector<uint16_t> values = {500, 2000, 1000};
    list.add_entries(3, doc_ids.data(),
                     reinterpret_cast<const uint8_t*>(values.data()));

    auto result = list.prune_and_keep_doc_ids(2);

    ASSERT_EQ(result.size(), 2);
    std::ranges::sort(result);
    ASSERT_EQ(result[0], 2);
    ASSERT_EQ(result[1], 3);
}

TEST(InvertedList, move_constructor) {
    amaiss::InvertedList list(amaiss::U32);
    amaiss::idx_t doc_id = 42;
    float value = 1.5F;
    list.add_entries(1, &doc_id, reinterpret_cast<const uint8_t*>(&value));

    amaiss::InvertedList moved(std::move(list));

    ASSERT_EQ(moved.get_doc_ids().size(), 1);
    ASSERT_EQ(moved.get_doc_ids()[0], 42);
}

// ArrayInvertedLists tests
TEST(ArrayInvertedLists, constructor) {
    amaiss::ArrayInvertedLists lists(10, amaiss::U32);
    ASSERT_EQ(lists.get_n_term(), 10);
    ASSERT_EQ(lists.get_element_size(), amaiss::U32);
    ASSERT_EQ(lists.size(), 10);
}

TEST(ArrayInvertedLists, add_entries_single_term) {
    amaiss::ArrayInvertedLists lists(5, amaiss::U32);
    std::vector<amaiss::idx_t> doc_ids = {1, 2, 3};
    std::vector<float> values = {1.0F, 2.0F, 3.0F};

    lists.add_entries(2, 3, doc_ids.data(),
                      reinterpret_cast<const uint8_t*>(values.data()));

    ASSERT_EQ(lists[2].get_doc_ids().size(), 3);
    ASSERT_EQ(lists[0].get_doc_ids().size(), 0);
    ASSERT_EQ(lists[1].get_doc_ids().size(), 0);
}

TEST(ArrayInvertedLists, add_entries_multiple_terms) {
    amaiss::ArrayInvertedLists lists(3, amaiss::U32);

    amaiss::idx_t doc1 = 10;
    float val1 = 1.0F;
    lists.add_entries(0, 1, &doc1, reinterpret_cast<const uint8_t*>(&val1));

    amaiss::idx_t doc2 = 20;
    float val2 = 2.0F;
    lists.add_entries(1, 1, &doc2, reinterpret_cast<const uint8_t*>(&val2));

    amaiss::idx_t doc3 = 30;
    float val3 = 3.0F;
    lists.add_entries(2, 1, &doc3, reinterpret_cast<const uint8_t*>(&val3));

    ASSERT_EQ(lists[0].get_doc_ids()[0], 10);
    ASSERT_EQ(lists[1].get_doc_ids()[0], 20);
    ASSERT_EQ(lists[2].get_doc_ids()[0], 30);
}

TEST(ArrayInvertedLists, add_entries_out_of_range_throws) {
    amaiss::ArrayInvertedLists lists(5, amaiss::U32);
    amaiss::idx_t doc_id = 1;
    float value = 1.0F;

    ASSERT_THROW(lists.add_entries(5, 1, &doc_id,
                                   reinterpret_cast<const uint8_t*>(&value)),
                 std::invalid_argument);

    ASSERT_THROW(lists.add_entries(100, 1, &doc_id,
                                   reinterpret_cast<const uint8_t*>(&value)),
                 std::invalid_argument);
}

TEST(ArrayInvertedLists, add_entry_single) {
    amaiss::ArrayInvertedLists lists(3, amaiss::U32);
    float value = 5.0F;

    lists.add_entry(1, 42, reinterpret_cast<const uint8_t*>(&value));

    ASSERT_EQ(lists[1].get_doc_ids().size(), 1);
    ASSERT_EQ(lists[1].get_doc_ids()[0], 42);
}

TEST(ArrayInvertedLists, operator_bracket_const) {
    amaiss::ArrayInvertedLists lists(3, amaiss::U32);
    amaiss::idx_t doc_id = 1;
    float value = 1.0F;
    lists.add_entries(0, 1, &doc_id, reinterpret_cast<const uint8_t*>(&value));

    const auto& const_lists = lists;
    ASSERT_EQ(const_lists[0].get_doc_ids().size(), 1);
}

TEST(ArrayInvertedLists, iterator) {
    amaiss::ArrayInvertedLists lists(3, amaiss::U32);

    int count = 0;
    for (auto& list : lists) {
        (void)list;
        count++;
    }
    ASSERT_EQ(count, 3);
}

TEST(ArrayInvertedLists, const_iterator) {
    amaiss::ArrayInvertedLists lists(3, amaiss::U32);

    const auto& const_lists = lists;
    int count = 0;
    for (const auto& list : const_lists) {
        (void)list;
        count++;
    }
    ASSERT_EQ(count, 3);
}

// build_inverted_lists tests
TEST(ArrayInvertedLists, build_inverted_lists_empty_vectors) {
    amaiss::SparseVectorsConfig config{.element_size = amaiss::U32,
                                       .dimension = 10};
    amaiss::SparseVectors vectors(config);

    auto invlists = amaiss::ArrayInvertedLists::build_inverted_lists(
        10, amaiss::U32, &vectors);

    ASSERT_NE(invlists, nullptr);
    ASSERT_EQ(invlists->get_n_term(), 10);
    ASSERT_EQ(invlists->get_element_size(), amaiss::U32);
    for (size_t i = 0; i < 10; ++i) {
        ASSERT_TRUE((*invlists)[i].get_doc_ids().empty());
    }
}

TEST(ArrayInvertedLists, build_inverted_lists_single_doc_single_term) {
    amaiss::SparseVectorsConfig config{.element_size = amaiss::U32,
                                       .dimension = 5};
    amaiss::SparseVectors vectors(config);

    std::vector<amaiss::term_t> indices = {2};
    float value = 3.5F;
    std::vector<uint8_t> weights(
        reinterpret_cast<uint8_t*>(&value),
        reinterpret_cast<uint8_t*>(&value) + sizeof(float));
    vectors.add_vector(indices, weights);

    auto invlists = amaiss::ArrayInvertedLists::build_inverted_lists(
        5, amaiss::U32, &vectors);

    ASSERT_EQ(invlists->size(), 5);
    // Only term 2 should have an entry
    ASSERT_EQ((*invlists)[2].get_doc_ids().size(), 1);
    ASSERT_EQ((*invlists)[2].get_doc_ids()[0], 0);  // doc_id = 0

    // Other terms should be empty
    ASSERT_TRUE((*invlists)[0].get_doc_ids().empty());
    ASSERT_TRUE((*invlists)[1].get_doc_ids().empty());
    ASSERT_TRUE((*invlists)[3].get_doc_ids().empty());
    ASSERT_TRUE((*invlists)[4].get_doc_ids().empty());
}

TEST(ArrayInvertedLists, build_inverted_lists_single_doc_multiple_terms) {
    amaiss::SparseVectorsConfig config{.element_size = amaiss::U32,
                                       .dimension = 5};
    amaiss::SparseVectors vectors(config);

    std::vector<amaiss::term_t> indices = {0, 2, 4};
    std::vector<float> values = {1.0F, 2.0F, 3.0F};
    std::vector<uint8_t> weights(reinterpret_cast<uint8_t*>(values.data()),
                                 reinterpret_cast<uint8_t*>(values.data()) +
                                     values.size() * sizeof(float));
    vectors.add_vector(indices, weights);

    auto invlists = amaiss::ArrayInvertedLists::build_inverted_lists(
        5, amaiss::U32, &vectors);

    // Terms 0, 2, 4 should have doc 0
    ASSERT_EQ((*invlists)[0].get_doc_ids().size(), 1);
    ASSERT_EQ((*invlists)[0].get_doc_ids()[0], 0);
    ASSERT_EQ((*invlists)[2].get_doc_ids().size(), 1);
    ASSERT_EQ((*invlists)[2].get_doc_ids()[0], 0);
    ASSERT_EQ((*invlists)[4].get_doc_ids().size(), 1);
    ASSERT_EQ((*invlists)[4].get_doc_ids()[0], 0);

    // Terms 1, 3 should be empty
    ASSERT_TRUE((*invlists)[1].get_doc_ids().empty());
    ASSERT_TRUE((*invlists)[3].get_doc_ids().empty());
}

TEST(ArrayInvertedLists, build_inverted_lists_multiple_docs_same_term) {
    amaiss::SparseVectorsConfig config{.element_size = amaiss::U32,
                                       .dimension = 3};
    amaiss::SparseVectors vectors(config);

    // Doc 0: term 1
    std::vector<amaiss::term_t> indices1 = {1};
    float val1 = 1.0F;
    std::vector<uint8_t> weights1(
        reinterpret_cast<uint8_t*>(&val1),
        reinterpret_cast<uint8_t*>(&val1) + sizeof(float));
    vectors.add_vector(indices1, weights1);

    // Doc 1: term 1
    std::vector<amaiss::term_t> indices2 = {1};
    float val2 = 2.0F;
    std::vector<uint8_t> weights2(
        reinterpret_cast<uint8_t*>(&val2),
        reinterpret_cast<uint8_t*>(&val2) + sizeof(float));
    vectors.add_vector(indices2, weights2);

    // Doc 2: term 1
    std::vector<amaiss::term_t> indices3 = {1};
    float val3 = 3.0F;
    std::vector<uint8_t> weights3(
        reinterpret_cast<uint8_t*>(&val3),
        reinterpret_cast<uint8_t*>(&val3) + sizeof(float));
    vectors.add_vector(indices3, weights3);

    auto invlists = amaiss::ArrayInvertedLists::build_inverted_lists(
        3, amaiss::U32, &vectors);

    // Term 1 should have all 3 docs
    ASSERT_EQ((*invlists)[1].get_doc_ids().size(), 3);
    auto doc_ids = (*invlists)[1].get_doc_ids();
    std::ranges::sort(doc_ids);
    ASSERT_EQ(doc_ids[0], 0);
    ASSERT_EQ(doc_ids[1], 1);
    ASSERT_EQ(doc_ids[2], 2);

    // Other terms should be empty
    ASSERT_TRUE((*invlists)[0].get_doc_ids().empty());
    ASSERT_TRUE((*invlists)[2].get_doc_ids().empty());
}

TEST(ArrayInvertedLists, build_inverted_lists_multiple_docs_different_terms) {
    amaiss::SparseVectorsConfig config{.element_size = amaiss::U32,
                                       .dimension = 4};
    amaiss::SparseVectors vectors(config);

    // Doc 0: terms 0, 1
    std::vector<amaiss::term_t> indices1 = {0, 1};
    std::vector<float> vals1 = {1.0F, 2.0F};
    std::vector<uint8_t> weights1(reinterpret_cast<uint8_t*>(vals1.data()),
                                  reinterpret_cast<uint8_t*>(vals1.data()) +
                                      vals1.size() * sizeof(float));
    vectors.add_vector(indices1, weights1);

    // Doc 1: terms 1, 2
    std::vector<amaiss::term_t> indices2 = {1, 2};
    std::vector<float> vals2 = {3.0F, 4.0F};
    std::vector<uint8_t> weights2(reinterpret_cast<uint8_t*>(vals2.data()),
                                  reinterpret_cast<uint8_t*>(vals2.data()) +
                                      vals2.size() * sizeof(float));
    vectors.add_vector(indices2, weights2);

    // Doc 2: terms 2, 3
    std::vector<amaiss::term_t> indices3 = {2, 3};
    std::vector<float> vals3 = {5.0F, 6.0F};
    std::vector<uint8_t> weights3(reinterpret_cast<uint8_t*>(vals3.data()),
                                  reinterpret_cast<uint8_t*>(vals3.data()) +
                                      vals3.size() * sizeof(float));
    vectors.add_vector(indices3, weights3);

    auto invlists = amaiss::ArrayInvertedLists::build_inverted_lists(
        4, amaiss::U32, &vectors);

    // Term 0: doc 0
    ASSERT_EQ((*invlists)[0].get_doc_ids().size(), 1);
    ASSERT_EQ((*invlists)[0].get_doc_ids()[0], 0);

    // Term 1: docs 0, 1
    ASSERT_EQ((*invlists)[1].get_doc_ids().size(), 2);
    auto term1_docs = (*invlists)[1].get_doc_ids();
    std::ranges::sort(term1_docs);
    ASSERT_EQ(term1_docs[0], 0);
    ASSERT_EQ(term1_docs[1], 1);

    // Term 2: docs 1, 2
    ASSERT_EQ((*invlists)[2].get_doc_ids().size(), 2);
    auto term2_docs = (*invlists)[2].get_doc_ids();
    std::ranges::sort(term2_docs);
    ASSERT_EQ(term2_docs[0], 1);
    ASSERT_EQ(term2_docs[1], 2);

    // Term 3: doc 2
    ASSERT_EQ((*invlists)[3].get_doc_ids().size(), 1);
    ASSERT_EQ((*invlists)[3].get_doc_ids()[0], 2);
}

TEST(ArrayInvertedLists, build_inverted_lists_uint8_element_size) {
    amaiss::SparseVectorsConfig config{.element_size = amaiss::U8,
                                       .dimension = 3};
    amaiss::SparseVectors vectors(config);

    std::vector<amaiss::term_t> indices = {0, 2};
    std::vector<uint8_t> weights = {100, 200};
    vectors.add_vector(indices, weights);

    auto invlists = amaiss::ArrayInvertedLists::build_inverted_lists(
        3, amaiss::U8, &vectors);

    ASSERT_EQ(invlists->get_element_size(), amaiss::U8);
    ASSERT_EQ((*invlists)[0].get_doc_ids().size(), 1);
    ASSERT_EQ((*invlists)[2].get_doc_ids().size(), 1);
    ASSERT_TRUE((*invlists)[1].get_doc_ids().empty());
}

TEST(ArrayInvertedLists, build_inverted_lists_uint16_element_size) {
    amaiss::SparseVectorsConfig config{.element_size = amaiss::U16,
                                       .dimension = 3};
    amaiss::SparseVectors vectors(config);

    std::vector<amaiss::term_t> indices = {1};
    uint16_t value = 1000;
    std::vector<uint8_t> weights(
        reinterpret_cast<uint8_t*>(&value),
        reinterpret_cast<uint8_t*>(&value) + sizeof(uint16_t));
    vectors.add_vector(indices, weights);

    auto invlists = amaiss::ArrayInvertedLists::build_inverted_lists(
        3, amaiss::U16, &vectors);

    ASSERT_EQ(invlists->get_element_size(), amaiss::U16);
    ASSERT_EQ((*invlists)[1].get_doc_ids().size(), 1);
    ASSERT_EQ((*invlists)[1].get_doc_ids()[0], 0);
}

TEST(ArrayInvertedLists, build_inverted_lists_preserves_values) {
    amaiss::SparseVectorsConfig config{.element_size = amaiss::U32,
                                       .dimension = 3};
    amaiss::SparseVectors vectors(config);

    std::vector<amaiss::term_t> indices = {1};
    float value = 42.5F;
    std::vector<uint8_t> weights(
        reinterpret_cast<uint8_t*>(&value),
        reinterpret_cast<uint8_t*>(&value) + sizeof(float));
    vectors.add_vector(indices, weights);

    auto invlists = amaiss::ArrayInvertedLists::build_inverted_lists(
        3, amaiss::U32, &vectors);

    const auto& codes = (*invlists)[1].get_codes();
    ASSERT_EQ(codes.size(), sizeof(float));
    float stored = *reinterpret_cast<const float*>(codes.data());
    ASSERT_FLOAT_EQ(stored, 42.5F);
}
