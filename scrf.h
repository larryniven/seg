#ifndef SCRF_H
#define SCRF_H

#include <vector>
#include <string>
#include <tuple>
#include <limits>
#include <ostream>
#include <iostream>
#include <unordered_map>
#include <map>
#include "ebt/ebt.h"
#include "scrf/fst.h"
#include "scrf/lm.h"
#include "scrf/lattice.h"
#include "la/la.h"
#include "scrf/ilat.h"

namespace scrf {

    struct feat_t {
        std::unordered_map<std::string, std::vector<real>> class_vec;
    };

    struct param_t {
        std::unordered_map<std::string, la::vector<real>> class_vec;
    };

    feat_t to_feat(param_t f);
    param_t to_param(feat_t f);

    void iadd(param_t& p1, param_t const& p2);
    void isub(param_t& p1, param_t const& p2);

    param_t& operator-=(param_t& p1, param_t const& p2);
    param_t& operator+=(param_t& p1, param_t const& p2);
    param_t& operator*=(param_t& p1, real c);

    real norm(param_t const& p);
    real dot(param_t const& p1, param_t const& p2);

    param_t load_param(std::istream& is);
    param_t load_param(std::string filename);

    void save_param(param_t const& param, std::ostream& os);
    void save_param(param_t const& param, std::string filename);

    void const_step_update_momentum(param_t& theta, param_t grad,
        param_t& update, real momentum, real step_size);

    void adagrad_update(param_t& theta, param_t const& grad,
        param_t& accu_grad_sq, real step_size);

    struct scrf_feature {

        virtual ~scrf_feature();

        virtual void operator()(
            feat_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const = 0;

    };

    struct scrf_weight {

        virtual ~scrf_weight();

        virtual real operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const = 0;

    };

    template <class base_fst, class weight_func_t,
        class feature_func_t, class cost_func_t = weight_func_t>
    struct scrf_fst {
        using fst_type = base_fst;
        using vertex = typename base_fst::vertex;
        using edge = typename base_fst::edge;

        std::shared_ptr<fst_type> fst;
        std::shared_ptr<weight_func_t> weight_func;
        std::shared_ptr<feature_func_t> feature_func;
        std::shared_ptr<feature_func_t> cost_func;

        std::vector<vertex> vertices() const
        {
            return fst->vertices();
        }

        std::vector<edge> edges() const
        {
            return fst->edges();
        }

        vertex head(edge const& e) const
        {
            return fst->head(e);
        }

        vertex tail(edge const& e) const
        {
            return fst->tail(e);
        }

        std::vector<edge> in_edges(vertex const& v) const
        {
            return fst->in_edges(v);
        }

        std::vector<edge> out_edges(vertex const& v) const
        {
            return fst->out_edges(v);
        }

        real weight(edge const& e) const
        {
            return (*weight_func)(*fst, e);
        }

        std::string input(edge const& e) const
        {
            return fst->input(e);
        }

        std::string output(edge const& e) const
        {
            return fst->output(e);
        }

        std::vector<vertex> initials() const
        {
            return fst->initials();
        }

        std::vector<vertex> finals() const
        {
            return fst->finals();
        }

        void feature(feat_t& f, edge const& e) const
        {
            (*feature_func)(f, *fst, e);
        }

        void cost(edge const& e) const
        {
            return (*cost_func)(e);
        }

    };

    struct scrf_t
        : public scrf_fst<fst::composed_fst<lattice::fst, lm::fst>, scrf_weight, scrf_feature> {

        std::vector<vertex> topo_order;

    };

    std::vector<std::tuple<int, int>> topo_order(scrf_t const& scrf);

    fst::path<scrf_t> shortest_path(scrf_t const& s,
        std::vector<std::tuple<int, int>> const& order);

    struct loss_func {

        virtual ~loss_func();

        virtual double loss() = 0;
        virtual param_t param_grad() = 0;

    };

    namespace first_order {

        struct param_t {
            std::vector<la::vector<real>> class_vec;
        };

        void iadd(param_t& p1, param_t const& p2);
        void isub(param_t& p1, param_t const& p2);

        param_t& operator-=(param_t& p1, param_t const& p2);
        param_t& operator+=(param_t& p1, param_t const& p2);
        param_t& operator*=(param_t& p1, real c);

        real norm(param_t const& p);
        real dot(param_t const& p1, param_t const& p2);

        param_t load_param(std::istream& is);
        param_t load_param(std::string filename);

        void save_param(param_t const& param, std::ostream& os);
        void save_param(param_t const& param, std::string filename);

        void const_step_update_momentum(param_t& theta, param_t const& grad,
            param_t& update, double momentum, double step_size);

        void adagrad_update(param_t& theta, param_t const& grad,
            param_t& accu_grad_sq, double step_size);

        struct scrf_feature {
            virtual ~scrf_feature();

            virtual void operator()(
                param_t& feat, ilat::fst const& fst, int e) const = 0;
        };

        struct feat_dim_alloc {

            std::vector<int> order_dim;
            std::vector<int> const& labels;

            feat_dim_alloc(std::vector<int> const& labels);

            int alloc(int order, int dim);
        };

        struct scrf_weight {

            virtual ~scrf_weight();

            virtual double operator()(ilat::fst const& fst, int e) const = 0;

        };

        struct scrf_t {
            using vertex = int;
            using edge = int;

            std::shared_ptr<ilat::fst> fst;
            std::shared_ptr<scrf_weight> weight_func;
            std::shared_ptr<scrf_feature> feature_func;
            std::shared_ptr<scrf_weight> cost_func;

            std::vector<vertex> topo_order;

            std::vector<vertex> vertices() const;
            std::vector<edge> edges() const;
            vertex head(int e) const;
            vertex tail(int e) const;
            std::vector<edge> in_edges(int v) const;
            std::vector<edge> out_edges(int v) const;
            double weight(int e) const;
            int input(int e) const;
            int output(int e) const;
            std::vector<vertex> initials() const;
            std::vector<vertex> finals() const;

            void feature(param_t& f, int e) const;
            double cost(int e) const;
        };

        fst::path<scrf_t> shortest_path(scrf_t const& s,
            std::vector<int> const& order);

        struct loss_func {

            virtual ~loss_func();

            virtual double loss() = 0;
            virtual param_t param_grad() = 0;

        };

    }

}

#endif
