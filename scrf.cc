#include "seg/scrf.h"
#include <istream>
#include <fstream>
#include <cassert>
#include "ebt/ebt.h"
#include "opt/opt.h"
#include "la/la.h"

namespace scrf {

    std::pair<int, int> get_dim(std::string feat)
    {
        std::vector<std::string> parts = ebt::split(feat, ":");
        int start_dim = -1;
        int end_dim = -1;
        if (parts.size() == 2) {
            std::vector<std::string> indices = ebt::split(parts.back(), "-");
            start_dim = std::stoi(indices.at(0));
            end_dim = std::stoi(indices.at(1));
        }

        return std::make_pair(start_dim, end_dim);
    }

}
