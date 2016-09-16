#ifndef ILAT_H
#define ILAT_H

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
        double weight;
        int input;
        int output;
    };

    bool operator==(vertex_data const& v1, vertex_data const& v2);
    bool operator==(edge_data const& e1, edge_data const& e2);

    struct fst_data {
        std::string name;

        std::shared_ptr<std::unordered_map<std::string, int>> symbol_id;
        std::shared_ptr<std::vector<std::string>> id_symbol;

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

        std::vector<std::unordered_map<int, std::vector<int>>> in_edges_map;
        std::vector<std::unordered_map<int, std::vector<int>>> out_edges_map;

        std::vector<std::vector<std::pair<std::string, std::string>>> vertex_attrs;
        std::vector<std::vector<std::pair<std::string, std::string>>> edge_attrs;

        std::vector<std::vector<double>> feats;
    };

    void add_vertex(fst_data& data, int v, vertex_data v_data);
    void add_edge(fst_data& data, int e, edge_data e_data);

    /*
     * The class `fst_data` is separated instead of inlined in `fst`,
     * because we want to separate data (`fst_data`) that can be manipulated
     * and interface (`fst`) for accessing the data.
     *
     * We make `data` a `shared_ptr` to avoid copying everything when `fst`
     * is copied.
     *
     */
    struct fst
        : public ::fst::fst<int, int, int>
        , public ::fst::timed<int> {

        using vertex = int;
        using edge = int;
        using symbol = int;

        std::shared_ptr<fst_data> data;

        virtual std::vector<int> const& vertices() const override;
        virtual std::vector<int> const& edges() const override;
        virtual double weight(int e) const override;
        virtual std::vector<int> const& in_edges(int v) const override;
        virtual std::vector<int> const& out_edges(int v) const override;
        virtual int tail(int e) const override;
        virtual int head(int e) const override;
        virtual std::vector<int> const& initials() const override;
        virtual std::vector<int> const& finals() const override;
        virtual int const& input(int e) const override;
        virtual int const& output(int e) const override;

        virtual long time(int v) const override;

        virtual std::unordered_map<int, std::vector<int>> const& in_edges_map(int v) const;
        virtual std::unordered_map<int, std::vector<int>> const& out_edges_map(int v) const;
    };

    /*
     * Only acceptors can be loaded as of now.
     *
     */
    fst load_lattice(std::istream& is,
        std::unordered_map<std::string, int> const& symbol_id);

    fst add_eps_loops(fst f, int label=0);

    struct ilat_path_maker
        : public ::fst::path_maker<fst> {

        virtual std::shared_ptr<fst> operator()(
            std::vector<int> const& edges, fst const& f) const override;
    };

    fst load_arpa_lm(std::istream& is,
        std::unordered_map<std::string, int> const& symbol_id);

    fst load_arpa_lm(std::string filename,
        std::unordered_map<std::string, int> const& symbol_id);

    /*
     * We allow multiple ways to implement a pair of `fst`.
     * The class `lazy_pair` is used for lazy composition, and
     * `composed_pair` is used for an already composed fst.
     *
     * The two functions `fst1()` and `fst2()` will be useful
     * for extracting features.
     *
     */
    struct pair_fst
        : public ::fst::fst<std::tuple<int, int>,
            std::tuple<int, int>, int>
        , public ::fst::timed<std::tuple<int, int>> {

        using vertex = std::tuple<int, int>;
        using edge = std::tuple<int, int>;
        using symbol = int;

        virtual ilat::fst const& fst1() const = 0;
        virtual ilat::fst const& fst2() const = 0;

    };

    struct lazy_pair_mode1
        : public pair_fst {

        using vertex = std::tuple<int, int>;
        using edge = std::tuple<int, int>;
        using symbol = int;

        ilat::fst fst1_;
        ilat::fst fst2_;

        mutable std::shared_ptr<std::vector<vertex>> vertices_cache;
        mutable std::shared_ptr<std::vector<edge>> edges_cache;
        mutable std::shared_ptr<std::vector<vertex>> initials_cache;
        mutable std::shared_ptr<std::vector<vertex>> finals_cache;
        mutable std::shared_ptr<vertex> in_edges_vertex;
        mutable std::shared_ptr<std::vector<edge>> in_edges_cache;
        mutable std::shared_ptr<vertex> out_edges_vertex;
        mutable std::shared_ptr<std::vector<edge>> out_edges_cache;

        lazy_pair_mode1(ilat::fst fst1, ilat::fst fst2);

        virtual std::vector<vertex> const& vertices() const override;
        virtual std::vector<edge> const& edges() const override;
        virtual double weight(edge e) const override;
        virtual std::vector<edge> const& in_edges(vertex v) const override;
        virtual std::vector<edge> const& out_edges(vertex v) const override;
        virtual vertex tail(edge e) const override;
        virtual vertex head(edge e) const override;
        virtual std::vector<vertex> const& initials() const override;
        virtual std::vector<vertex> const& finals() const override;
        virtual int const& input(edge e) const override;
        virtual int const& output(edge e) const override;
        virtual long time(vertex v) const override;

        virtual ilat::fst const& fst1() const;
        virtual ilat::fst const& fst2() const;
    };

    struct lazy_pair_mode2
        : public pair_fst {

        using vertex = std::tuple<int, int>;
        using edge = std::tuple<int, int>;
        using symbol = int;

        ilat::fst fst1_;
        ilat::fst fst2_;

        mutable std::shared_ptr<std::vector<vertex>> vertices_cache;
        mutable std::shared_ptr<std::vector<edge>> edges_cache;
        mutable std::shared_ptr<std::vector<vertex>> initials_cache;
        mutable std::shared_ptr<std::vector<vertex>> finals_cache;
        mutable std::shared_ptr<vertex> in_edges_vertex;
        mutable std::shared_ptr<std::vector<edge>> in_edges_cache;
        mutable std::shared_ptr<vertex> out_edges_vertex;
        mutable std::shared_ptr<std::vector<edge>> out_edges_cache;

        lazy_pair_mode2(ilat::fst fst1, ilat::fst fst2);

        virtual std::vector<vertex> const& vertices() const override;
        virtual std::vector<edge> const& edges() const override;
        virtual double weight(edge e) const override;
        virtual std::vector<edge> const& in_edges(vertex v) const override;
        virtual std::vector<edge> const& out_edges(vertex v) const override;
        virtual vertex tail(edge e) const override;
        virtual vertex head(edge e) const override;
        virtual std::vector<vertex> const& initials() const override;
        virtual std::vector<vertex> const& finals() const override;
        virtual int const& input(edge e) const override;
        virtual int const& output(edge e) const override;
        virtual long time(vertex v) const override;

        virtual ilat::fst const& fst1() const;
        virtual ilat::fst const& fst2() const;
    };

    struct pair_data {
        using vertex = std::tuple<int, int>;
        using edge = std::tuple<int, int>;

        std::vector<vertex> vertices;
        std::vector<edge> edges;
        std::unordered_map<vertex, std::vector<edge>> in_edges;
        std::unordered_map<vertex, std::vector<edge>> out_edges;
        std::vector<vertex> initials;
        std::vector<vertex> finals;

        std::vector<edge> empty;
    };

    struct composed_pair
        : public pair_fst {

        ilat::fst fst1_;
        ilat::fst fst2_;

        std::shared_ptr<pair_data> data;

        virtual std::vector<vertex> const& vertices() const override;
        virtual std::vector<edge> const& edges() const override;
        virtual double weight(edge e) const override;
        virtual std::vector<edge> const& in_edges(vertex v) const override;
        virtual std::vector<edge> const& out_edges(vertex v) const override;
        virtual vertex tail(edge e) const override;
        virtual vertex head(edge e) const override;
        virtual std::vector<vertex> const& initials() const override;
        virtual std::vector<vertex> const& finals() const override;
        virtual int const& input(edge e) const override;
        virtual int const& output(edge e) const override;
        virtual long time(vertex v) const override;

        virtual ilat::fst const& fst1() const;
        virtual ilat::fst const& fst2() const;
    };

    struct pair_fst_path_maker
        : public ::fst::path_maker<pair_fst> {

        virtual std::shared_ptr<pair_fst> operator()(std::vector<pair_fst::edge> const& edges,
            pair_fst const& f) const;

    };

}

namespace fst {

    template <>
    struct edge_trait<int> {
        static int null;
    };

    template <>
    struct edge_trait<std::tuple<int, int>> {
        static std::tuple<int, int> null;
    };

    template <>
    struct symbol_trait<int> {
        static int eps;
    };

}

#endif
