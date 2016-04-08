#include "scrf/lattice.h"
#include "ebt/ebt.h"
#include <algorithm>
#include <cassert>

namespace lattice {

    bool operator==(edge_data const& e1, edge_data const& e2)
    {
        return e1.label == e2.label && e1.tail == e2.tail
            && e1.head == e2.head && e1.weight == e2.weight;
    }

    void add_vertex(fst_data& data, int v, ilat::vertex_data v_data)
    {
        if (!ebt::in(v, data.vertex_set)) {
            data.vertex_set.insert(v);
            data.vertex_indices.push_back(v);

            int size = std::max<int>(v + 1, data.vertices.size());

            data.vertices.resize(size);
            data.vertices[v] = v_data;
            data.in_edges.resize(size);
            data.out_edges.resize(size);
            data.in_edges_map.resize(size);
            data.out_edges_map.resize(size);
        } else {
            assert(data.vertices[v] == v_data);
        }
    }

    void add_edge(fst_data& data, int e, edge_data e_data)
    {
        assert(ebt::in(e_data.head, data.vertex_set));
        assert(ebt::in(e_data.tail, data.vertex_set));

        if (!ebt::in(e, data.edge_set)) {
            data.edge_set.insert(e);
            data.edge_indices.push_back(e);

            int size = std::max<int>(e + 1, data.edges.size());

            data.edges.resize(size);
            data.edges[e] = e_data;
            data.in_edges[e_data.head].push_back(e);
            data.out_edges[e_data.tail].push_back(e);
            data.in_edges_map[e_data.head][e_data.label].push_back(e);
            data.out_edges_map[e_data.tail][e_data.label].push_back(e);
            data.attrs.resize(size);
            data.feats.resize(size);
        } else {
            assert(data.edges[e] == e_data);
        }
    }

    std::vector<int> const& fst::vertices() const
    {
        return data->vertex_indices;
    }

    std::vector<int> const& fst::edges() const
    {
        return data->edge_indices;
    }

    double fst::weight(int e) const
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

    std::vector<int> const& fst::initials() const
    {
        return data->initials;
    }

    std::vector<int> const& fst::finals() const
    {
        return data->finals;
    }

    std::string const& fst::input(int e) const
    {
        return data->edges.at(e).label;
    }

    std::string const& fst::output(int e) const
    {
        return data->edges.at(e).label;
    }

    long fst::time(int v) const
    {
        return data->vertices.at(v).time;
    }

    std::unordered_map<std::string, std::vector<int>> const&
    fst::in_edges_map(int v) const
    {
        return data->in_edges_map.at(v);
    }
    
    std::unordered_map<std::string, std::vector<int>> const&
    fst::out_edges_map(int v) const
    {
        return data->out_edges_map.at(v);
    }

    fst load_lattice(std::istream& is)
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

            add_vertex(result, v, ilat::vertex_data { std::stoi(attr.at("time")) });
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
            add_edge(result, e, edge_data { tail, head, weight, label });

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

    fst add_eps_loops(fst f, std::string label)
    {
        fst_data& data = *(f.data);

        for (int i = 0; i < data.vertices.size(); ++i) {
            int e = int(data.edges.size());
            add_edge(data, e, edge_data {i, i, 0, label});
        }

        return f;
    }

}
