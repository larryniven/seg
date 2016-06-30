#ifndef UTIL_H
#define UTIL_H

#include <unordered_map>
#include <vector>
#include "scrf/segcost.h"

namespace util {

    std::unordered_map<std::string, int> load_label_id(std::string filename);

    std::vector<std::string> load_label_seq(std::istream& is);

    std::vector<segcost::segment<int>> load_segments(std::istream& is,
        std::unordered_map<std::string, int> const& label_id);

    std::vector<segcost::segment<std::string>> load_segments(std::istream& is);

    std::vector<std::string> load_labels(std::istream& is);

}

#endif
