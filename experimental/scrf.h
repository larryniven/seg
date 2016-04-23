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

    template <class edge, class vector>
    struct with_feature {
        virtual void feature(vector& f, edge e) const = 0;
    };

    template <class edge>
    struct with_cost {
        virtual double cost(edge e) const = 0;
    };

    template <class edge, class vector>
    struct with_frame_grad {
        virtual void frame_grad(std::vector<std::vector<double>>& f,
            vector const& param, edge e) const = 0;
    };

    template <class vertex, class edge, class symbol, class vector>
    struct scrf
        : public fst::fst<vertex, edge, symbol>
        , public fst::timed<vertex>
        , public fst::with_topo_order<vertex>
        , public with_feature<edge, vector>
        , public with_cost<edge> {
    };

    template <class fst, class vector>
    struct scrf_feature {

        virtual ~scrf_feature()
        {}

        virtual void operator()(vector& f, fst const& a,
            typename fst::edge e) const = 0;

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

    void adagrad_update(dense_vec& theta, dense_vec const& grad,
        dense_vec& accu_grad_sq, double step_size);

    void rmsprop_update(dense_vec& theta, dense_vec const& grad,
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

    std::unordered_map<std::string, int> load_label_id(std::string filename);

    template <class vector>
    struct loss_func {

        virtual ~loss_func()
        {}

        virtual double loss() const = 0;
        virtual vector param_grad() const = 0;

    };

    template <class vector>
    struct loss_func_with_frame_grad
        : public loss_func<vector> {

        virtual void frame_grad(std::vector<std::vector<double>>& grad,
            vector const& param) const = 0;
    };

    std::pair<int, int> get_dim(std::string feat);

}

#endif
