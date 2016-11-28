#include "seg/util.h"
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

    std::vector<int> load_label_seq(std::istream& is,
        std::unordered_map<std::string, int> const& label_id)
    {
        std::string line;
        std::getline(is, line);

        std::vector<std::string> parts;

        if (is) {
            parts = ebt::split(line);
            parts.pop_back();
        }

        std::vector<int> result;
        for (auto& s: parts) {
            result.push_back(label_id.at(s));
        }

        return result;
    }

    std::vector<segcost::segment<int>> load_segments(std::istream& is,
        std::unordered_map<std::string, int> const& label_id, int subsample_freq)
    {
        std::string line;

        std::vector<segcost::segment<int>> result;

        // get rid of the name
        std::getline(is, line);

        while (std::getline(is, line) && line != ".") {
            auto parts = ebt::split(line);

            result.push_back(segcost::segment<int> {
                 .start_time = std::lrint(std::stoi(parts[0]) / double(subsample_freq)),
                 .end_time = std::lrint(std::stoi(parts[1]) / double(subsample_freq)),
                 .label = label_id.at(parts[2])
            });
        }

        return result;
    }

    std::vector<segcost::segment<std::string>> load_segments(std::istream& is)
    {
        std::string line;

        std::vector<segcost::segment<std::string>> result;

        // get rid of the name
        std::getline(is, line);

        while (std::getline(is, line) && line != ".") {
            auto parts = ebt::split(line);

            result.push_back(segcost::segment<std::string> {
                 .start_time = std::stoi(parts[0]),
                 .end_time = std::stoi(parts[1]),
                 .label = parts[2]
            });
        }

        return result;
    }

    std::vector<std::string> load_labels(std::istream& is)
    {
        std::string line;

        std::getline(is, line);

        if (!is) {
            return std::vector<std::string>();
        }

        std::vector<std::string> parts = ebt::split(line);

        if (ebt::startswith(parts.back(), "(") && ebt::endswith(parts.back(), ")")) {
            parts.pop_back();
        }

        return parts;
    }

    void batch_indices::open(std::string filename)
    {
        std::ifstream ifs { filename };
    
        std::string line;
        std::getline(ifs, line);
    
        if (!ifs) {
            std::cerr << "unable to open " << filename << std::endl;
            exit(1);
        }
    
        stream.open(line);
    
        if (!stream) {
            std::cerr << "unable to open " << line << std::endl;
            exit(1);
        }
    
        while (std::getline(ifs, line)) {
            pos.push_back(std::stoul(line));
        }
    
        ifs.close();
    }

    std::ifstream& batch_indices::at(int i)
    {
        stream.seekg(pos.at(i));
        return stream;
    }

}
