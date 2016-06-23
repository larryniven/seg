#ifndef UTIL_H
#define UTIL_H

#include <unordered_map>
#include <vector>

namespace util {

    std::unordered_map<std::string, int> load_label_id(std::string filename);

    std::vector<std::string> load_label_seq(std::istream& is);

}

#endif
