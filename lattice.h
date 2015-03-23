#ifndef LATTICE_H
#define LATTICE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>

namespace lattice {

    struct vertex_data {
        long time;
    };

    struct edge_data {
        std::string label;
        int tail;
        int head;
    };

    struct fst_data {
        int initial;
        int final;
    
        std::vector<vertex_data> vertices;
        std::vector<edge_data> edges;
        std::vector<std::vector<int>> in_edges;
        std::vector<std::vector<int>> out_edges;

        std::vector<std::unordered_map<std::string, std::vector<int>>> in_edges_map;
        std::vector<std::unordered_map<std::string, std::vector<int>>> out_edges_map;
    };

    struct fst {
        using vertex_type = int;
        using edge_type = int;

        std::shared_ptr<fst_data> data;

        std::vector<int> vertices() const;
        std::vector<int> edges() const;
        double weight(int e) const;
        std::vector<int> const& in_edges(int v) const;
        std::vector<int> const& out_edges(int v) const;
        int tail(int e) const;
        int head(int e) const;
        int initial() const;
        int final() const;
        std::string const& input(int e) const;
        std::string const& output(int e) const;

        std::unordered_map<std::string, std::vector<int>> const& in_edges_map(int v) const;
        std::unordered_map<std::string, std::vector<int>> const& out_edges_map(int v) const;
    };

    fst make_fst(
        std::vector<std::vector<double>> acoustics,
        std::unordered_set<std::string> phone_set);

    fst load_lattice(std::istream& is);

    fst add_eps_loops(fst fst);

}

#endif
