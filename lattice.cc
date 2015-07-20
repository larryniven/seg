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

    void add_vertex(fst_data& data, int v, long time)
    {
        assert(v <= data.vertices.size());

        if (v < data.vertices.size()) {
            assert(data.vertices.at(v).time == time);
        } else if (v == data.vertices.size()) {
            data.vertices.push_back(vertex_data { time });
            data.in_edges.resize(data.vertices.size());
            data.out_edges.resize(data.vertices.size());
            data.in_edges_map.resize(data.vertices.size());
            data.out_edges_map.resize(data.vertices.size());
        }
    }

    void add_edge(fst_data& data, int e, std::string label, int tail, int head, real weight)
    {
        assert(head < data.vertices.size());
        assert(tail < data.vertices.size());

        assert(e <= data.edges.size());

        if (e < data.edges.size()) {
            assert((edge_data {label, tail, head, weight} == data.edges.at(e)));
        } else if (e == data.edges.size()) {
            data.edges.push_back(edge_data {label, tail, head, weight});
            data.in_edges[head].push_back(e);
            data.out_edges[tail].push_back(e);
            data.out_edges_map.at(tail)[label].push_back(e);
            data.in_edges_map.at(head)[label].push_back(e);
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

    std::string const& fst::input(int e) const
    {
        return data->edges.at(e).label;
    }

    std::string const& fst::output(int e) const
    {
        return data->edges.at(e).label;
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
            auto pairs = ebt::split(parts[1], ",");
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

            std::unordered_map<std::string, std::string> attr;
            auto pairs = ebt::split(parts[2], ",");
            for (auto& p: pairs) {
                auto pair = ebt::split(p, "=");
                attr[pair[0]] = pair[1];
            }

            real weight = 0;
            if (ebt::in(std::string("weight"), attr)) {
                weight = std::stod(attr.at("weight"));
            }

            std::string label = attr.at("label");

            int e = int(result.edges.size());
            add_edge(result, e, label, tail, head, weight);

            for (auto& p: attr) {
                if (p.first == "weight" || p.first == "label") {
                    continue;
                }

                result.features[e](p.first) = std::stod(p.second);
            }

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
        auto& data = *(f.data);

        std::unordered_set<int> initial_set { data.initials.begin(), data.initials.end() };
        std::unordered_set<int> final_set { data.finals.begin(), data.finals.end() };

        for (int i = 0; i < data.vertices.size(); ++i) {
            if (ebt::in(i, initial_set) || ebt::in(i, final_set)) {
                continue;
            }

            int e = int(data.edges.size());
            add_edge(data, e, label, i, i, 0);
        }

        return f;
    }

    std::vector<int> topo_order(lattice::fst const& fst)
    {
        std::vector<int> order;
        std::unordered_set<int> order_set;
        std::vector<int> stack = fst.initials();
        std::unordered_set<int> traversed { stack.begin(), stack.end() };

        std::vector<int> path;

        while (stack.size() > 0) {
            int u = stack.back();
            stack.pop_back();

            // TODO: alternatively make in_edges return an unordered_set?
            auto u_in_edges = fst.in_edges(u);
            std::unordered_set<int> u_in_edge_set { u_in_edges.begin(), u_in_edges.end() };

            while (path.size() > 0 && !ebt::in(path.back(), u_in_edge_set)) {
                int v = path.back();
                if (!ebt::in(v, order_set)) {
                    order.push_back(v);
                    order_set.insert(v);
                }
                path.pop_back();
            }

            path.push_back(u);

            for (int e: fst.out_edges(u)) {
                int v = fst.head(e);

                if (!ebt::in(v, traversed)) {
                    stack.push_back(v);
                    traversed.insert(v);
                }
            }
        }

        while (path.size() > 0) {
            int v = path.back();
            if (!ebt::in(v, order_set)) {
                order.push_back(v);
                order_set.insert(v);
            }
            order.push_back(v);
            path.pop_back();
        }

        std::reverse(order.begin(), order.end());

        return order;
    }

}
