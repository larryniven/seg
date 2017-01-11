#ifndef UTIL_H
#define UTIL_H

#include <unordered_map>
#include <vector>
#include "seg/segcost.h"
#include <fstream>

namespace util {

    std::vector<segcost::segment<int>> load_segments(std::istream& is,
        std::unordered_map<std::string, int> const& label_id, int subsample_freq=1);

    std::vector<segcost::segment<std::string>> load_segments(std::istream& is);

}

#endif
