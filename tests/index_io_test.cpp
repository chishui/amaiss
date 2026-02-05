#include "amaiss/io/index_io.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "amaiss/brutal_index.h"
#include "amaiss/index.h"
#include "amaiss/io/buffered_io.h"
#include "amaiss/io/io.h"
#include "amaiss/seismic_index.h"
#include "amaiss/seismic_scalar_quantized_index.h"
#include "amaiss/sparse_vectors.h"
#include "amaiss/types.h"

namespace {

// Mock Index that implements both Index and IndexIO for testing
class MockIndex : public amaiss::Index, public amaiss::IndexIO {
public:
    static constexpr std::array<char, 4> name = {'M', 'O', 'C', 'K'};

    explicit MockIndex(int dim = 0) : Index(dim) {}

    std::array<char, 4> id() const override { return name; }

    void add(amaiss::idx_t /*n*/, const amaiss::idx_t* /*indptr*/,
             const amaiss::term_t* /*indices*/,
             const float* /*values*/) override {}

    const amaiss::SparseVectors* get_vectors() const override {
        return nullptr;
    }

    // IndexIO implementation
    void write_index(amaiss::IOWriter* io_writer) override {
        io_writer->write(&test_data_, sizeof(int), 1);
        size_t size = test_string_.size();
        io_writer->write(&size, sizeof(size_t), 1);
        io_writer->write(test_string_.data(), sizeof(char), size);
    }

    void read_index(amaiss::IOReader* io_reader) override {
        io_reader->read(&test_data_, sizeof(int), 1);
        size_t size = 0;
        io_reader->read(&size, sizeof(size_t), 1);
        test_string_.resize(size);
        io_reader->read(test_string_.data(), sizeof(char), size);
    }

    void set_test_data(int data) { test_data_ = data; }
    int get_test_data() const { return test_data_; }

    void set_test_string(const std::string& str) { test_string_ = str; }
    const std::string& get_test_string() const { return test_string_; }

private:
    int test_data_ = 0;
    std::string test_string_;
};

}  // namespace

// Test write_index with BufferedIOWriter
TEST(IndexIO, WriteIndexBasic) {
    MockIndex index(128);
    index.set_test_data(42);
    index.set_test_string("hello");

    amaiss::BufferedIOWriter writer;
    amaiss::write_index(&index, &writer);

    // Verify something was written
    ASSERT_GT(writer.size(), 0);

    // Verify header: first 4 bytes should be the fourcc
    const auto& data = writer.data();
    uint32_t written_fourcc = 0;
    std::memcpy(&written_fourcc, data.data(), sizeof(uint32_t));
    ASSERT_EQ(written_fourcc, amaiss::fourcc(MockIndex::name));

    // Verify dimension is written after fourcc
    int written_dim = 0;
    std::memcpy(&written_dim, data.data() + sizeof(uint32_t), sizeof(int));
    ASSERT_EQ(written_dim, 128);
}

// Test write_index throws for non-IndexIO index
TEST(IndexIO, WriteIndexThrowsForNonIndexIO) {
    // Create a minimal Index that doesn't implement IndexIO
    class NonSerializableIndex : public amaiss::Index {
    public:
        NonSerializableIndex() : Index(10) {}
        std::array<char, 4> id() const override { return {'N', 'O', 'I', 'O'}; }
        void add(amaiss::idx_t, const amaiss::idx_t*, const amaiss::term_t*,
                 const float*) override {}
        const amaiss::SparseVectors* get_vectors() const override {
            return nullptr;
        }
    };

    NonSerializableIndex index;
    amaiss::BufferedIOWriter writer;

    ASSERT_THROW(amaiss::write_index(&index, &writer), std::runtime_error);
}

// Test read_index throws for unknown index type
TEST(IndexIO, ReadIndexThrowsForUnknownType) {
    // Create a buffer with an unknown fourcc
    std::vector<uint8_t> buffer(sizeof(uint32_t) + sizeof(int));
    uint32_t unknown_fourcc = 0xDEADBEEF;
    int dimension = 64;
    std::memcpy(buffer.data(), &unknown_fourcc, sizeof(uint32_t));
    std::memcpy(buffer.data() + sizeof(uint32_t), &dimension, sizeof(int));

    amaiss::BufferedIOReader reader(buffer);
    ASSERT_THROW(amaiss::read_index(&reader), std::runtime_error);
}

// Test roundtrip with SeismicIndex (real index that supports serialization)
TEST(IndexIO, RoundtripSeismicIndex) {
    // Create and write a SeismicIndex
    auto* original = new amaiss::SeismicIndex(256);

    amaiss::BufferedIOWriter writer;
    amaiss::write_index(original, &writer);

    // Read it back
    amaiss::BufferedIOReader reader(writer.data());
    amaiss::Index* loaded = amaiss::read_index(&reader);

    ASSERT_NE(loaded, nullptr);
    ASSERT_EQ(loaded->get_dimension(), 256);
    ASSERT_EQ(loaded->id(), original->id());

    delete original;
    delete loaded;
}

// Test write_index throws for BrutalIndex (doesn't implement IndexIO)
TEST(IndexIO, WriteIndexThrowsForBrutalIndex) {
    auto* index = new amaiss::BrutalIndex(512);

    amaiss::BufferedIOWriter writer;
    ASSERT_THROW(amaiss::write_index(index, &writer), std::runtime_error);

    delete index;
}

// Test roundtrip with SeismicScalarQuantizedIndex
TEST(IndexIO, RoundtripSeismicScalarQuantizedIndex) {
    auto* original = new amaiss::SeismicScalarQuantizedIndex(1024);

    amaiss::BufferedIOWriter writer;
    amaiss::write_index(original, &writer);

    amaiss::BufferedIOReader reader(writer.data());
    amaiss::Index* loaded = amaiss::read_index(&reader);

    ASSERT_NE(loaded, nullptr);
    ASSERT_EQ(loaded->get_dimension(), 1024);
    ASSERT_EQ(loaded->id(), original->id());

    delete original;
    delete loaded;
}

// Test write_index with empty writer
TEST(IndexIO, WriteIndexEmptyWriter) {
    auto* index = new amaiss::SeismicIndex(64);

    amaiss::BufferedIOWriter writer;
    ASSERT_EQ(writer.size(), 0);

    amaiss::write_index(index, &writer);
    ASSERT_GT(writer.size(), 0);

    delete index;
}

// Test read_index with small dimension
TEST(IndexIO, ReadIndexSmallDimension) {
    // Use a real SeismicIndex to create a valid buffer
    auto* original = new amaiss::SeismicIndex(32);

    amaiss::BufferedIOWriter writer;
    amaiss::write_index(original, &writer);

    amaiss::BufferedIOReader reader(writer.data());
    amaiss::Index* loaded = amaiss::read_index(&reader);

    ASSERT_NE(loaded, nullptr);
    ASSERT_EQ(loaded->get_dimension(), 32);

    delete original;
    delete loaded;
}

// Test BufferedIOWriter and BufferedIOReader work correctly together
TEST(IndexIO, BufferedIOWriterReaderIntegration) {
    amaiss::BufferedIOWriter writer;

    // Write various data types
    int int_val = 12345;
    float float_val = 3.14159F;
    std::vector<uint8_t> bytes = {1, 2, 3, 4, 5};

    writer.write(&int_val, sizeof(int), 1);
    writer.write(&float_val, sizeof(float), 1);
    writer.write(bytes.data(), sizeof(uint8_t), bytes.size());

    // Read back
    amaiss::BufferedIOReader reader(writer.data());

    int read_int = 0;
    float read_float = 0.0F;
    std::vector<uint8_t> read_bytes(5);

    reader.read(&read_int, sizeof(int), 1);
    reader.read(&read_float, sizeof(float), 1);
    reader.read(read_bytes.data(), sizeof(uint8_t), 5);

    ASSERT_EQ(read_int, int_val);
    ASSERT_FLOAT_EQ(read_float, float_val);
    ASSERT_EQ(read_bytes, bytes);
}

// Test multiple write/read cycles
TEST(IndexIO, MultipleWriteReadCycles) {
    for (int i = 0; i < 3; ++i) {
        auto* index = new amaiss::SeismicIndex(64 * (i + 1));

        amaiss::BufferedIOWriter writer;
        amaiss::write_index(index, &writer);

        amaiss::BufferedIOReader reader(writer.data());
        amaiss::Index* loaded = amaiss::read_index(&reader);

        ASSERT_EQ(loaded->get_dimension(), 64 * (i + 1));

        delete index;
        delete loaded;
    }
}
