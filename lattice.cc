#include "scrf/lattice.h"
#include "ebt/ebt.h"
#include <algorithm>

namespace lattice {

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

        std::getline(is, line);

        int min_time = std::numeric_limits<int>::max();
        std::unordered_set<int> initials;

        int max_time = std::numeric_limits<int>::min();
        std::unordered_set<int> finals;

        while (std::getline(is, line) && line != ".") {
            auto parts = ebt::split(line);

            int tail = std::stoi(parts[3]);
            int head = std::stoi(parts[4]);

            if (std::max(tail, head) >= result.vertices.size()) {
                result.vertices.resize(std::max<int>(tail, head) + 1);
            }

            result.vertices.at(tail).time = std::stoi(parts[0]) / 1e5;
            result.vertices.at(head).time = std::stoi(parts[1]) / 1e5;

            real weight = 0;
            if (parts.size() > 5) {
                weight = std::stod(parts.at(5));
            }

            std::string label = parts[2] == "<eps>" ? "<spe>" : parts[2];

            edge_data e_data { .label = label, .tail = tail, .head = head, .weight = weight };

            result.edges.push_back(e_data);

            if (std::max(tail, head) >= result.in_edges.size()) {
                result.in_edges.resize(std::max(tail, head) + 1);
                result.out_edges.resize(std::max(tail, head) + 1);
                result.in_edges_map.resize(std::max(tail, head) + 1);
                result.out_edges_map.resize(std::max(tail, head) + 1);
            }

            int e = int(result.edges.size()) - 1;

            result.in_edges[head].push_back(e);
            result.out_edges[tail].push_back(e);

            result.out_edges_map.at(tail)[label].push_back(e);
            result.in_edges_map.at(head)[label].push_back(e);

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

            edge_data e_data { .label = label,
                .tail = i, .head = i, .weight = 0};
            data.edges.push_back(e_data);

            data.in_edges[i].push_back(data.edges.size() - 1);
            data.in_edges_map[i][label].push_back(data.edges.size() - 1);

            data.out_edges[i].push_back(data.edges.size() - 1);
            data.out_edges_map[i][label].push_back(data.edges.size() - 1);
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

        auto is_parent = [&](int u, int v) {
            for (int e: fst.out_edges(u)) {
                if (v == fst.head(e)) {
                    return true;
                }
            }

            return false;
        };

        while (stack.size() > 0) {
            int u = stack.back();
            stack.pop_back();

            while (path.size() > 0 && !is_parent(path.back(), u)) {
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
