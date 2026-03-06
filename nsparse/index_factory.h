#ifndef INDEX_FACTORY_H
#define INDEX_FACTORY_H

#include "nsparse/index.h"
namespace nsparse {

Index* index_factory(int dimension, const char* description);
}

#endif  // INDEX_FACTORY_H