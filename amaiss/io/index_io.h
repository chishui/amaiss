#ifndef INDEX_IO_H
#define INDEX_IO_H

#include "amaiss/index.h"
#include "amaiss/io/io.h"
namespace amaiss {

void write_index(Index* index, char* filename);
Index* read_index(char* filename);

}  // namespace amaiss

#endif  // INDEX_IO_H