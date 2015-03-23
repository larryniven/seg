#ifndef LM_H
#define LM_H

#include <vector>
#include <string>
#include <unordered_map>
#include <istream>
#include <memory>

namespace lm {

    struct edge_data {
        int tail;
        int head;
        double weight;
        std::string label;
    };

    struct fst_data {
        int vertices;
        std::vector<edge_data> edges;
        std::vector<std::vector<int>> in_edges;
        std::vector<std::unordered_map<std::string, std::vector<int>>> in_edges_map;
        std::vector<std::vector<int>> out_edges;
        std::vector<std::unordered_map<std::string, std::vector<int>>> out_edges_map;

        int initial;
        int final;
    };

    struct fst {
        using vertex_type = int;
        using edge_type = int;

        std::shared_ptr<fst_data> data;

        std::vector<int> vertices() const;
        std::vector<int> edges() const;
        double weight(int e) const;
        std::vector<int> const& in_edges(int v) const;
        std::unordered_map<std::string, std::vector<int>> const& in_edges_map(int v) const;
        std::vector<int> const& out_edges(int v) const;
        std::unordered_map<std::string, std::vector<int>> const& out_edges_map(int v) const;
        
        int tail(int e) const;
        int head(int e) const;
        int initial() const;
        int final() const;
        std::string const& input(int e) const;
        std::string const& output(int e) const;
    };

    fst load_arpa_lm(std::string filename);
    fst load_arpa_lm(std::istream& filename);

}

#endif
