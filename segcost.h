#ifndef SEG_COST_H
#define SEG_COST_H

#include <vector>

namespace segcost {

    template <class symbol>
    struct segment {
        long start_time;
        long end_time;
        symbol label;
    };

    template <class symbol>
    struct cost {
        
        virtual ~cost()
        {}

        virtual double operator()(std::vector<segment<symbol>> const& gold_edges,
            segment<symbol> const& e) const = 0;

    };

    template <class symbol>
    struct hit_cost
        : public cost<symbol> {

        virtual double operator()(std::vector<segment<symbol>> const& gold_edges,
            segment<symbol> const& e) const override;

    };

    template <class symbol>
    struct overlap_cost
        : public cost<symbol> {

        std::vector<symbol> sils;

        overlap_cost();
        overlap_cost(std::vector<symbol> sils);

        virtual double operator()(std::vector<segment<symbol>> const& gold_edges,
            segment<symbol> const& e) const override;

    };

    template <class symbol>
    struct overlap_portion_cost
        : public cost<symbol> {

        std::vector<symbol> sils;

        overlap_portion_cost();
        overlap_portion_cost(std::vector<symbol> sils);

        virtual double operator()(std::vector<segment<symbol>> const& gold_edges,
            segment<symbol> const& e) const override;

    };

    template <class symbol>
    struct cover_cost
        : public cost<symbol> {

        std::vector<symbol> sils;

        cover_cost();
        cover_cost(std::vector<symbol> sils);

        virtual double operator()(std::vector<segment<symbol>> const& gold_edges,
            segment<symbol> const& e) const override;

    };

}

#include "seg/segcost.h"

#endif
