#ifndef COST_H
#define COST_H

#include <vector>

namespace cost {

    template <class symbol>
    struct segment {
        long start_time;
        long end_time;
        symbol label;
    };

    template <class symbol>
    struct cost_t {
        
        virtual ~cost_t()
        {}

        virtual double operator()(std::vector<segment<symbol>> const& gold_edges,
            segment<symbol> const& e) const = 0;

    };

    template <class symbol>
    struct overlap_cost
        : public cost_t<symbol> {

        std::vector<symbol> sils;

        overlap_cost();
        overlap_cost(std::vector<symbol> sils);

        virtual double operator()(std::vector<segment<symbol>> const& gold_edges,
            segment<symbol> const& e) const override;

    };

}

#include "seg/cost-impl.h"

#endif
