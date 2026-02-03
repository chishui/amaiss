#ifndef INDEX_FACTORY_H
#define INDEX_FACTORY_H

#include "amaiss/index.h"
namespace amaiss {

Index* index_factory(int dimension, const char* description);
}

#endif  // INDEX_FACTORY_H