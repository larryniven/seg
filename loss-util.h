#ifndef LOSS_UTIL_H
#define LOSS_UTIL_H

#include <unordered_map>

namespace seg {

    template <class fst_type>
    struct risk_func {
        virtual ~risk_func()
        {};

        virtual double operator()(fst_type const& f,
            typename fst_type::edge e) const = 0;

        virtual void accumulate_grad(double g, fst_type const& f,
            typename fst_type::edge e)
        {}

        virtual void grad()
        {}

    };

    template <class fst_type>
    struct forward_exp_risk {

        using vertex = typename fst_type::vertex;

        std::unordered_map<vertex, double> extra;

        std::shared_ptr<risk_func<fst_type>> risk;

        forward_exp_risk(std::shared_ptr<risk_func<fst_type>> risk);

        void merge(fst_type const& f, std::vector<vertex> order,
            std::unordered_map<vertex, double> const& forward_log_sum);

    };

    template <class fst_type>
    struct backward_exp_risk {

        using vertex = typename fst_type::vertex;

        std::unordered_map<vertex, double> extra;

        std::shared_ptr<risk_func<fst_type>> risk;

        backward_exp_risk(std::shared_ptr<risk_func<fst_type>> risk);

        void merge(fst_type const& f, std::vector<vertex> order,
            std::unordered_map<vertex, double> const& forward_log_sum);

    };

}

#include "seg/loss-util-impl.h"

#endif
