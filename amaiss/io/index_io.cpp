#include "amaiss/io/index_io.h"

#include <stdexcept>

#include "amaiss/brutal_index.h"
#include "amaiss/id_map_index.h"
#include "amaiss/io/file_io.h"
#include "amaiss/seismic_index.h"
#include "amaiss/seismic_scalar_quantized_index.h"

namespace amaiss {

namespace {
constexpr uint32_t BRUT = fourcc(BrutalIndex::name);
constexpr uint32_t SEIS = fourcc(SeismicIndex::name);
constexpr uint32_t SESQ = fourcc(SeismicScalarQuantizedIndex::name);
constexpr uint32_t IDMP = fourcc(IDMapIndex::name);

void write_header(Index* index, IOWriter* io_writer) {
    // write index type
    auto id_val = fourcc(index->id());
    io_writer->write(&id_val, sizeof(uint32_t), 1);
    // write dimension
    auto dimension = index->get_dimension();
    io_writer->write(&dimension, sizeof(int), 1);
}

Index* read_header(IOReader* io_reader) {
    uint32_t id_val = 0;
    io_reader->read(&id_val, sizeof(uint32_t), 1);
    int dimension = 0;
    io_reader->read(&dimension, sizeof(int), 1);
    switch (id_val) {
        case BRUT:
            return new BrutalIndex(dimension);
        case SEIS:
            return new SeismicIndex(dimension);
        case SESQ:
            return new SeismicScalarQuantizedIndex(dimension);
        case IDMP:
            return new IDMapIndex();
        default:
            throw std::runtime_error("Unknown index type");
    }
}
}  // namespace
void write_index(Index* index, IOWriter* io_writer) {
    auto* index_io = dynamic_cast<IndexIO*>(index);
    if (index_io == nullptr) {
        throw std::runtime_error("Index does not support serialization");
    }
    // write header
    write_header(index, io_writer);
    // write index customized payload
    index_io->write_index(io_writer);
    io_writer->close();
}

void write_index(Index* index, char* filename) {
    FileIOWriter writer(filename);
    write_index(index, &writer);
}

Index* read_index(IOReader* io_reader) {
    Index* index = read_header(io_reader);
    auto* index_io = dynamic_cast<IndexIO*>(index);
    if (index_io == nullptr) {
        throw std::runtime_error("Index does not support serialization");
    }
    index_io->read_index(io_reader);
    io_reader->close();
    return index;
}

Index* read_index(char* filename) {
    FileIOReader reader(filename);
    return read_index(&reader);
}
}  // namespace amaiss
