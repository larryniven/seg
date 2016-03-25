#ifndef SCRF_FEAT_UTIL_H
#define SCRF_FEAT_UTIL_H

#include <string>
#include <unordered_map>

namespace scrf {

    std::unordered_map<std::string, int> load_label_id(std::string filename);

}

#endif
