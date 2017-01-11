#ifndef LOSS_UTIL_H
#define LOSS_UTIL_H

namespace seg {

    template <class fst_type>
    struct forward_exp_score {

        using vertex = typename fst_type::vertex;

        std::unordered_map<vertex, double> extra;

        void merge(fst_type const& f, std::vector<vertex> order,
            double logZ,
            std::unordered_map<vertex, double> const& forward_log_sum);

    };

    template <class fst_type>
    struct backward_exp_score {

        using vertex = typename fst_type::vertex;

        std::unordered_map<vertex, double> extra;

        void merge(fst_type const& f, std::vector<vertex> order,
            double logZ,
            std::unordered_map<vertex, double> const& backward_log_sum);

    };

}

#include "seg/loss-util-impl.h"

#endif
