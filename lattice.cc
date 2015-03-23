#include "scrf/lattice.h"
#include "ebt/ebt.h"

namespace lattice {

    fst make_fst_data(
        std::vector<std::vector<double>> acoustics,
        std::unordered_set<std::string> phone_set)
    {
        fst_data result;

        result.initial = 0;
        result.final = acoustics.size();

        for (int i = 0; i <= acoustics.size(); ++i) {
            vertex_data v_data;
            v_data.time = i;
            result.vertices.push_back(v_data);
        }

        for (int i = 0; i <= acoustics.size(); ++i) {
            for (auto& p: phone_set) {
                if (p == "<eps>") {
                    continue;
                }

                for (int j = 1; j <= 20 && i + j <= acoustics.size(); ++j) {
                    int tail = i;
                    int head = i + j;

                    result.edges.push_back(edge_data { .label = p,
                        .tail = tail, .head = head });

                    if (std::max(tail, head) >= int(result.in_edges.size()) - 1) {
                        result.in_edges.resize(std::max(tail, head) + 1);
                        result.out_edges.resize(std::max(tail, head) + 1);
                        result.in_edges_map.resize(std::max(tail, head) + 1);
                        result.out_edges_map.resize(std::max(tail, head) + 1);
                    }

                    result.in_edges[head].push_back(int(result.edges.size()) - 1);
                    result.out_edges[tail].push_back(int(result.edges.size()) - 1);

                    result.out_edges_map.at(tail)[p].push_back(result.edges.size() - 1);
                    result.in_edges_map.at(head)[p].push_back(result.edges.size() - 1);
                }
            }
        }

        fst f;
        f.data = std::make_shared<fst_data>(std::move(result));
    
        return f;
    }

    fst load_lattice(std::istream& is)
    {
        fst_data result;

        std::string line;

        std::getline(is, line);

        int final = 0;

        while (std::getline(is, line) && line != ".") {
            auto parts = ebt::split(line);

            int tail = std::stoi(parts[3]);
            int head = std::stoi(parts[4]);

            if (std::max(tail, head) >= result.vertices.size()) {
                result.vertices.resize(std::max<int>(tail, head) + 1);
            }

            result.vertices.at(tail).time = std::stoi(parts[0]) / 1e5;
            result.vertices.at(head).time = std::stoi(parts[1]) / 1e5;

            edge_data e_data { .label = parts[2], .tail = tail, .head = head };

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

            result.out_edges_map.at(tail)[parts[2]].push_back(e);
            result.in_edges_map.at(head)[parts[2]].push_back(e);

            final = std::max(final, head);
        }

        result.initial = 0;
        result.final = final;

        fst f;
        f.data = std::make_shared<fst_data>(std::move(result));

        return f;
    }

    fst add_eps_loops(fst f)
    {
        auto& data = *(f.data);

        for (int i = 0; i < data.vertices.size(); ++i) {
            edge_data e_data { .label = std::string("<eps>"),
                .tail = i, .head = i};
            data.edges.push_back(e_data);

            if (i >= data.in_edges.size()) {
                data.in_edges.resize(i + 1);
            }

            data.in_edges[i].push_back(data.edges.size() - 1);

            if (i >= data.out_edges.size()) {
                data.out_edges.resize(i + 1);
            }

            data.out_edges[i].push_back(data.edges.size() - 1);
        }

        return f;
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

    double fst::weight(int e) const
    {
        return 0;
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

    int fst::initial() const
    {
        return data->initial;
    }

    int fst::final() const
    {
        return data->final;
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
    
}
