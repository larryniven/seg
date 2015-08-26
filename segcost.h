#ifndef SEG_COST_H
#define SEG_COST_H

#include <vector>
#include "speech/speech.h"

namespace segcost {

    struct cost {

        virtual double operator()(std::vector<speech::segment> const& gold_edges,
            speech::segment const& e) const = 0;

    };

    struct overlap_cost
        : public cost {

        virtual double operator()(std::vector<speech::segment> const& gold_edges,
            speech::segment const& e) const override;

    };

}

#endif
