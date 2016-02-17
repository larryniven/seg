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
#include "scrf/seg.h"
#include "la/la.h"

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

    struct scrf_t
        : public seg::fst<fst::composed_fst<lattice::fst, lm::fst>, scrf_weight, scrf_feature> {

        std::vector<vertex_type> topo_order;

    };

    std::vector<std::tuple<int, int>> topo_order(scrf_t const& scrf);

    fst::path<scrf_t> shortest_path(scrf_t const& s,
        std::vector<std::tuple<int, int>> const& order);

    struct loss_func {

        virtual ~loss_func();

        virtual real loss() = 0;
        virtual param_t param_grad() = 0;

    };

}

#endif
