#ifndef ILAT_H
#define ILAT_H

#include "scrf/util.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include "ebt/ebt.h"
#include "scrf/fst.h"

namespace ilat {

    struct vertex_data {
        long time;
    };

    struct edge_data {
        int tail;
        int head;
        real weight;
        int label;
    };

    bool operator==(vertex_data const& v1, vertex_data const& v2);
    bool operator==(edge_data const& e1, edge_data const& e2);

    struct fst_data {
        std::string name;

        std::unordered_map<std::string, int> symbol_id;
        std::vector<std::string> id_symbol;

        std::vector<int> initials;
        std::vector<int> finals;
    
        std::vector<int> vertex_indices;
        std::vector<int> edge_indices;

        std::unordered_set<int> vertex_set;
        std::unordered_set<int> edge_set;

        std::vector<vertex_data> vertices;
        std::vector<edge_data> edges;

        std::vector<std::vector<int>> in_edges;
        std::vector<std::vector<int>> out_edges;

        std::vector<std::vector<std::pair<std::string, std::string>>> attrs;

        std::vector<std::vector<double>> feats;
    };

    void add_vertex(fst_data& data, int v, vertex_data v_data);
    void add_edge(fst_data& data, int e, edge_data e_data);

    struct fst
        : public ::fst::experimental::fst<int, int, int>
        , public ::fst::experimental::timed<int> {

        using vertex = int;
        using edge = int;
        using symbol = int;

        std::shared_ptr<fst_data> data;

        virtual std::vector<int> const& vertices() const override;
        virtual std::vector<int> const& edges() const override;
        virtual double weight(int e) const;
        virtual std::vector<int> const& in_edges(int v) const override;
        virtual std::vector<int> const& out_edges(int v) const override;
        virtual int tail(int e) const override;
        virtual int head(int e) const override;
        virtual std::vector<int> const& initials() const override;
        virtual std::vector<int> const& finals() const override;
        virtual int const& input(int e) const override;
        virtual int const& output(int e) const override;
        virtual long time(int v) const override;
    };

    fst load_lattice(std::istream& is,
        std::unordered_map<std::string, int> const& symbol_id);

    fst add_eps_loops(fst f, int label=0);

    struct ilat_path_maker
        : public ::fst::experimental::path_maker<ilat::fst> {

        virtual fst operator()(std::vector<int> const& edges, fst const& f) const override;
    };

}

#endif
