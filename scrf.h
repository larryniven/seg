#ifndef SCRF_H
#define SCRF_H

#include <vector>
#include <string>
#include <tuple>
#include <limits>
#include <ostream>
#include <iostream>
#include <unordered_map>
#include "ebt/ebt.h"
#include "seg/fst.h"
#include "la/la.h"
#include "seg/ilat.h"

namespace scrf {

    template <class fst>
    struct scrf_weight {

        virtual ~scrf_weight()
        {}

        virtual double operator()(fst const& f,
            typename fst::edge e) const = 0;

        virtual void accumulate_grad(double g, fst const& f,
            typename fst::edge e) const
        {}

        virtual void grad() const
        {}

    };

    std::pair<int, int> get_dim(std::string feat);

    template <class scrf_data>
    struct scrf_data_trait;

    template <class scrf_data>
    struct scrf_fst {

        using vertex = typename scrf_data_trait<scrf_data>::vertex;
        using edge = typename scrf_data_trait<scrf_data>::edge;
        using symbol = typename scrf_data_trait<scrf_data>::symbol;

        scrf_data const& data;

        std::vector<vertex> const& vertices() const;
        std::vector<edge> const& edges() const;
        vertex head(edge e) const;
        vertex tail(edge e) const;
        std::vector<edge> const& in_edges(vertex v) const;
        std::vector<edge> const& out_edges(vertex v) const;
        double weight(edge e) const;
        symbol const& input(edge e) const;
        symbol const& output(edge e) const;
        std::vector<vertex> const& initials() const;
        std::vector<vertex> const& finals() const;

        long time(vertex v) const;
        std::vector<vertex> const& topo_order() const;

    };

    template <class scrf_data>
    std::shared_ptr<typename scrf_data_trait<scrf_data>::base_fst> shortest_path(scrf_data const& data);

}

namespace scrf {

    template <class scrf_data>
    std::vector<typename scrf_data_trait<scrf_data>::vertex> const&
    scrf_fst<scrf_data>::vertices() const
    {
        return data.fst->vertices();
    }

    template <class scrf_data>
    std::vector<typename scrf_data_trait<scrf_data>::edge> const&
    scrf_fst<scrf_data>::edges() const
    {
        return data.fst->edges();
    }

    template <class scrf_data>
    typename scrf_data_trait<scrf_data>::vertex
    scrf_fst<scrf_data>::head(typename scrf_data_trait<scrf_data>::edge e) const
    {
        return data.fst->head(e);
    }

    template <class scrf_data>
    typename scrf_data_trait<scrf_data>::vertex
    scrf_fst<scrf_data>::tail(typename scrf_data_trait<scrf_data>::edge e) const
    {
        return data.fst->tail(e);
    }

    template <class scrf_data>
    std::vector<typename scrf_data_trait<scrf_data>::edge> const&
    scrf_fst<scrf_data>::in_edges(typename scrf_data_trait<scrf_data>::vertex v) const
    {
        return data.fst->in_edges(v);
    }

    template <class scrf_data>
    std::vector<typename scrf_data_trait<scrf_data>::edge> const&
    scrf_fst<scrf_data>::out_edges(typename scrf_data_trait<scrf_data>::vertex v) const
    {
        return data.fst->out_edges(v);
    }

    template <class scrf_data>
    double scrf_fst<scrf_data>::weight(typename scrf_data_trait<scrf_data>::edge e) const
    {
        return (*data.weight_func)(*data.fst, e);
    }

    template <class scrf_data>
    typename scrf_data_trait<scrf_data>::symbol const&
    scrf_fst<scrf_data>::input(typename scrf_data_trait<scrf_data>::edge e) const
    {
        return data.fst->input(e);
    }

    template <class scrf_data>
    typename scrf_data_trait<scrf_data>::symbol const&
    scrf_fst<scrf_data>::output(typename scrf_data_trait<scrf_data>::edge e) const
    {
        return data.fst->output(e);
    }

    template <class scrf_data>
    std::vector<typename scrf_data_trait<scrf_data>::vertex> const&
    scrf_fst<scrf_data>::initials() const
    {
        return data.fst->initials();
    }

    template <class scrf_data>
    std::vector<typename scrf_data_trait<scrf_data>::vertex> const&
    scrf_fst<scrf_data>::finals() const
    {
        return data.fst->finals();
    }

    template <class scrf_data>
    long scrf_fst<scrf_data>::time(typename scrf_data_trait<scrf_data>::vertex v) const
    {
        return data.fst->time(v);
    }

    template <class scrf_data>
    std::vector<typename scrf_data_trait<scrf_data>::vertex> const&
    scrf_fst<scrf_data>::topo_order() const
    {
        return *data.topo_order;
    }

    template <class scrf_data>
    std::shared_ptr<typename scrf_data_trait<scrf_data>::base_fst>
    shortest_path(scrf_data const& data)
    {
        scrf_fst<scrf_data> f { data };
        fst::forward_one_best<scrf_fst<scrf_data>> one_best;
        for (auto& v: f.initials()) {
            one_best.extra[v] = { fst::edge_trait<typename scrf_data_trait<scrf_data>::edge>::null, 0 };
        }
        one_best.merge(f, f.topo_order());
        std::vector<typename scrf_data_trait<scrf_data>::edge> edges = one_best.best_path(f);

        return typename scrf_data_trait<scrf_data>::path_maker()(edges, *data.fst);
    }

}

#endif
