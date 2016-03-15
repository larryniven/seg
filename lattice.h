#ifndef LATTICE_H
#define LATTICE_H

#include "scrf/util.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include "ebt/ebt.h"

namespace lattice {

    struct vertex_data {
        long time;
    };

    struct edge_data {
        int tail;
        int head;
        real weight;
        std::string label;
    };

    bool operator==(edge_data const& e1, edge_data const& e2);

    struct fst_data {
        std::string name;

        std::vector<int> initials;
        std::vector<int> finals;
    
        std::vector<vertex_data> vertices;
        std::vector<edge_data> edges;
        std::vector<std::vector<int>> in_edges;
        std::vector<std::vector<int>> out_edges;

        std::vector<std::unordered_map<std::string, std::vector<int>>> in_edges_map;
        std::vector<std::unordered_map<std::string, std::vector<int>>> out_edges_map;

        std::vector<std::vector<std::pair<std::string, std::string>>> attrs;

        std::vector<std::vector<double>> feats;
    };

    void add_vertex(fst_data& data, int v, long time);
    void add_edge(fst_data& data, int e, std::string label, int tail, int head, real weight);

    struct fst {
        using vertex = int;
        using edge = int;

        std::shared_ptr<fst_data> data;

        std::vector<int> vertices() const;
        std::vector<int> edges() const;
        real weight(int e) const;
        std::vector<int> const& in_edges(int v) const;
        std::vector<int> const& out_edges(int v) const;
        int tail(int e) const;
        int head(int e) const;
        std::vector<int> initials() const;
        std::vector<int> finals() const;
        std::string const& input(int e) const;
        std::string const& output(int e) const;
        long time(int v) const;

        std::unordered_map<std::string, std::vector<int>> const& in_edges_map(int v) const;
        std::unordered_map<std::string, std::vector<int>> const& out_edges_map(int v) const;
    };

    fst load_lattice(std::istream& is);

    fst load_path(std::istream& is);

    fst add_eps_loops(fst fst, std::string label="<eps>");

    std::vector<int> topo_order(lattice::fst const& fst);

}

#endif
