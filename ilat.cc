#include "scrf/ilat.h"
#include "ebt/ebt.h"
#include <algorithm>
#include <cassert>

namespace ilat {

    bool operator==(edge_data const& e1, edge_data const& e2)
    {
        return e1.label == e2.label && e1.tail == e2.tail
            && e1.head == e2.head && e1.weight == e2.weight;
    }

    void add_vertex(fst_data& data, int v, long time)
    {
        assert(v <= data.vertices.size());

        if (v < data.vertices.size()) {
            assert(data.vertices.at(v).time == time);
        } else if (v == data.vertices.size()) {
            data.vertices.push_back(vertex_data { time });
            data.in_edges.resize(data.vertices.size());
            data.out_edges.resize(data.vertices.size());
        }
    }

    void add_edge(fst_data& data, int e, int label, int tail, int head, real weight)
    {
        assert(head < data.vertices.size());
        assert(tail < data.vertices.size());

        assert(e <= data.edges.size());

        if (e < data.edges.size()) {
            assert((edge_data {tail, head, weight, label} == data.edges.at(e)));
        } else if (e == data.edges.size()) {
            data.edges.push_back(edge_data {tail, head, weight, label});
            data.in_edges[head].push_back(e);
            data.out_edges[tail].push_back(e);
            data.attrs.resize(std::max<int>(data.attrs.size(), e + 1));
            data.feats.resize(std::max<int>(data.feats.size(), e + 1));
        }
    }

    std::vector<int> fst::vertices() const
    {
        std::vector<int> result;

        for (int i = 0; i < data->vertices.size(); ++i) {
            result.push_back(i);
        }

        return result;
    }

    std::vector<int> fst::edges() const
    {
        std::vector<int> result;

        for (int i = 0; i < data->edges.size(); ++i) {
            result.push_back(i);
        }

        return result;
    }

    real fst::weight(int e) const
    {
        return data->edges.at(e).weight;
    }

    std::vector<int> const& fst::in_edges(int v) const
    {
        return data->in_edges.at(v);
    }

    std::vector<int> const& fst::out_edges(int v) const
    {
        return data->out_edges.at(v);
    }

    int fst::tail(int e) const
    {
        return data->edges.at(e).tail;
    }

    int fst::head(int e) const
    {
        return data->edges.at(e).head;
    }

    std::vector<int> fst::initials() const
    {
        return data->initials;
    }

    std::vector<int> fst::finals() const
    {
        return data->finals;
    }

    int fst::input(int e) const
    {
        return data->edges.at(e).label;
    }

    int fst::output(int e) const
    {
        return data->edges.at(e).label;
    }

    long fst::time(int v) const
    {
        return data->vertices.at(v).time;
    }

    fst load_lattice(std::istream& is, std::unordered_map<std::string, int> const& label_id)
    {
        fst_data result;

        std::string line;

        // read filename
        if (!std::getline(is, line)) {
            fst f;
            f.data = std::make_shared<fst_data>(result);
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

            std::unordered_map<std::string, std::string> attr;
            auto pairs = ebt::split(parts[1], ";");
            for (auto& p: pairs) {
                auto pair = ebt::split(p, "=");
                attr[pair[0]] = pair[1];
            }

            add_vertex(result, v, std::stoi(attr.at("time")));
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
            add_edge(result, e, label_id.at(label), tail, head, weight);

            result.attrs[e] = attr;
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

        fst f;
        f.data = std::make_shared<fst_data>(std::move(result));

        return f;
    }

}
