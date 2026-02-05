#ifndef INDEX_IO_H
#define INDEX_IO_H

#include "amaiss/index.h"
#include "amaiss/io/io.h"
namespace amaiss {

void write_index(Index* index, char* filename);
Index* read_index(char* filename);
void write_index(Index* index, IOWriter* io_writer);
Index* read_index(IOReader* io_reader);

}  // namespace amaiss

#endif  // INDEX_IO_H