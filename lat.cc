#include "seg/lat.h"
#include "ebt/ebt.h"
#include <algorithm>

namespace lat {

    ifst::fst load_lattice(std::istream& is, std::unordered_map<std::string, int> const& symbol_id)
    {
        ifst::fst_data result;

        result.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(symbol_id);
        
        std::vector<std::string> id_symbol;
        id_symbol.resize(symbol_id.size());
        for (auto& p: symbol_id) {
            id_symbol[p.second] = p.first;
        }
        result.id_symbol = std::make_shared<std::vector<std::string>>(id_symbol);

        std::string line;

        // read filename
        if (!std::getline(is, line)) {
            ifst::fst f;
            f.data = std::make_shared<ifst::fst_data>(result);
            return f;
        }

        result.name = line;

        int min_time = std::numeric_limits<int>::max();
        std::unordered_set<int> initials;

        int max_time = std::numeric_limits<int>::min();
        std::unordered_set<int> finals;

        while (std::getline(is, line) && line != "#") {
            if (line == ".") {
                std::cout << "\"#\" was not found between vertices and edges." << std::endl;
                exit(1);
            }

            auto parts = ebt::split(line);

            int v = std::stoi(parts[0]);

            std::unordered_map<std::string, std::string> attr_map;
            std::vector<std::pair<std::string, std::string>> attrs;
            auto pairs = ebt::split(parts[1], ";");
            for (auto& p: pairs) {
                auto pair = ebt::split(p, "=");
                attr_map[pair[0]] = pair[1];
                attrs.push_back(std::make_pair(pair[0], pair[1]));
            }

            ifst::add_vertex(result, v, ifst::vertex_data { std::stoi(attr_map.at("time")) });

            result.vertex_attrs[v] = attrs;
        }

        while (std::getline(is, line) && line != ".") {
            auto parts = ebt::split(line);

            int tail = std::stoi(parts[0]);
            int head = std::stoi(parts[1]);

            std::unordered_map<std::string, std::string> attr_map;
            std::vector<std::pair<std::string, std::string>> attr;
            std::vector<double> feats;

            auto pairs = ebt::split(parts[2], ";");
            for (auto& p: pairs) {
                auto pair = ebt::split(p, "=");

                if (pair[0] == "feat") {
                    auto parts = ebt::split(pair[1], ",");
                    feats.resize(parts.size());
                    std::transform(parts.begin(), parts.end(), feats.begin(),
                        [](std::string const& s) { return std::stod(s); });
                } else {
                    attr_map[pair[0]] = pair[1];
                    attr.push_back(std::make_pair(pair[0], pair[1]));
                }
            }

            double weight = 0;
            if (ebt::in(std::string("weight"), attr_map)) {
                weight = std::stod(attr_map.at("weight"));
            }

            std::string label = attr_map.at("label");

            int e = int(result.edges.size());
            ifst::add_edge(result, e, ifst::edge_data { tail, head, weight,
                symbol_id.at(label), symbol_id.at(label) });

            result.edge_attrs[e] = attr;
            result.feats[e] = feats;

            if (max_time < result.vertices.at(head).time) {
                max_time = result.vertices.at(head).time;
                finals.clear();
                finals.insert(head);
            } else if (max_time == result.vertices.at(head).time) {
                finals.insert(head);
            }

            if (result.vertices.at(tail).time < min_time) {
                min_time = result.vertices.at(tail).time;
                initials.clear();
                initials.insert(tail);
            } else if (min_time == result.vertices.at(tail).time) {
                initials.insert(tail);
            }
        }

        result.initials = std::vector<int> { initials.begin(), initials.end() };
        result.finals = std::vector<int> { finals.begin(), finals.end() };

        ifst::fst f;
        f.data = std::make_shared<ifst::fst_data>(std::move(result));

        return f;
    }

}
