#include "scrf/experimental/util.h"
#include <fstream>
#include "ebt/ebt.h"

namespace util {

    std::unordered_map<std::string, int> load_label_id(std::string filename)
    {
        std::unordered_map<std::string, int> result;
        std::string line;
        std::ifstream ifs { filename };

        result["<eps>"] = 0;
    
        int i = 1;
        while (std::getline(ifs, line)) {
            result[line] = i;
            ++i;
        }
    
        return result;
    }

    std::vector<std::string> load_label_seq(std::istream& is)
    {
        std::string line;
        std::getline(is, line);

        std::vector<std::string> parts;

        if (is) {
            parts = ebt::split(line);
            parts.pop_back();
        }

        return parts;
    }

}
