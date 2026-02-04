#ifndef IO_H
#define IO_H
#include <array>
#include <cstdint>

namespace amaiss {
class IOReader {
public:
    virtual ~IOReader() = default;
    virtual size_t read(void* ptr, size_t size, size_t nitems) = 0;
};

class IOWriter {
public:
    virtual ~IOWriter() = default;
    virtual void write(void* ptr, size_t size, size_t nitems) = 0;
};

class Serializable {
public:
    virtual ~Serializable() = default;
    virtual void serialize(IOWriter* writer) const = 0;
    virtual void deserialize(IOReader* reader) = 0;
};

class IndexIO {
public:
    virtual ~IndexIO() = default;
    virtual void write_index(IOWriter* io_writer) {};
    virtual void read_index(IOReader* io_reader) {};
};

constexpr uint32_t fourcc(const std::array<char, 4>& id) {
    return id[0] | id[1] << 8 | id[2] << 16 | id[3] << 24;
}
}  // namespace amaiss

#endif  // IO_H