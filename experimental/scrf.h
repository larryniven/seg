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
#include "scrf/experimental/fst.h"
#include "la/la.h"
#include "scrf/experimental/ilat.h"

namespace scrf {

    template <class fst, class vector>
    struct scrf_feature {

        virtual ~scrf_feature()
        {}

        virtual void operator()(vector& f, fst const& a,
            typename fst::edge e) const = 0;

    };

    template <class fst, class vector>
    struct scrf_feature_with_grad
        : public scrf_feature<fst, vector> {

        virtual void grad(vector const& g, fst const& a,
            typename fst::edge e) = 0;

    };

    template <class fst, class vector>
    struct scrf_feature_with_frame_grad
        : public scrf_feature<fst, vector> {

        virtual void frame_grad(std::vector<std::vector<double>>& grad,
            vector const& param, fst const& a,
            typename fst::edge e) const = 0;
    };

    template <class fst>
    struct scrf_weight {

        virtual ~scrf_weight()
        {}

        virtual double operator()(fst const& f,
            typename fst::edge e) const = 0;

    };

    struct dense_vec {
        std::vector<la::vector<double>> class_vec;
    };

    dense_vec load_dense_vec(std::istream& is);
    dense_vec load_dense_vec(std::string filename);

    void save_vec(dense_vec const& v, std::ostream& os);
    void save_vec(dense_vec const& v, std::string filename);

    double dot(dense_vec const& u, dense_vec const& v);
    void iadd(dense_vec& u, dense_vec const& v);
    void isub(dense_vec& u, dense_vec const& v);
    void imul(dense_vec& u, double c);

    void const_step_update(dense_vec& param, dense_vec const& grad,
        double step_size);

    void const_step_update_momentum(dense_vec& param, dense_vec const& grad,
        dense_vec& opt_data, double momentum, double step_size);

    void adagrad_update(dense_vec& param, dense_vec const& grad,
        dense_vec& accu_grad_sq, double step_size);

    void rmsprop_update(dense_vec& param, dense_vec const& grad,
        dense_vec& accu_grad_sq, double decay, double step_size);

    struct sparse_vec {
        std::unordered_map<std::string, la::vector<double>> class_vec;
    };

    sparse_vec load_sparse_vec(std::istream& is);
    sparse_vec load_sparse_vec(std::string filename);

    void save_vec(sparse_vec const& v, std::ostream& os);
    void save_vec(sparse_vec const& v, std::string filename);

    double dot(sparse_vec const& u, sparse_vec const& v);
    void iadd(sparse_vec& u, sparse_vec const& v);
    void isub(sparse_vec& u, sparse_vec const& v);
    void imul(sparse_vec& u, double c);

    void adagrad_update(sparse_vec& theta, sparse_vec const& grad,
        sparse_vec& accu_grad_sq, double step_size);

    void rmsprop_update(sparse_vec& theta, sparse_vec const& grad,
        sparse_vec& accu_grad_sq, double decay, double step_size);

    template <class vector>
    struct loss_func {

        virtual ~loss_func()
        {}

        virtual double loss() const = 0;
        virtual vector param_grad() const = 0;

    };

    template <class vector, class fst>
    struct loss_func_with_frame_grad
        : public loss_func<vector> {

        virtual void frame_grad(
            scrf_feature_with_frame_grad<fst, vector> const& feat_func,
            std::vector<std::vector<double>>& grad,
            vector const& param) const = 0;
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
        typename scrf_data_trait<scrf_data>::fst f { data };
        fst::forward_one_best<typename scrf_data_trait<scrf_data>::fst> one_best;
        for (auto& v: f.initials()) {
            one_best.extra[v] = { fst::edge_trait<typename scrf_data_trait<scrf_data>::edge>::null, 0 };
        }
        one_best.merge(f, f.topo_order());
        std::vector<typename scrf_data_trait<scrf_data>::edge> edges = one_best.best_path(f);

        return typename scrf_data_trait<scrf_data>::path_maker()(edges, *data.fst);
    }

}

#endif
