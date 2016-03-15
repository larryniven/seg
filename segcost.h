#ifndef SEG_COST_H
#define SEG_COST_H

#include <vector>
#include "speech/speech.h"

namespace segcost {

    struct cost {

        virtual ~cost();

        virtual double operator()(std::vector<speech::segment> const& gold_edges,
            speech::segment const& e) const = 0;

    };

    struct overlap_cost
        : public cost {

        virtual double operator()(std::vector<speech::segment> const& gold_edges,
            speech::segment const& e) const override;

    };

    namespace first_order {

        struct segment {
            long start_time;
            long end_time;
            int label;
        };

        struct cost {

            virtual ~cost();

            virtual double operator()(std::vector<segment> const& gold_edges,
                segment const& e) const = 0;

        };

        struct overlap_cost
            : public cost {

            std::vector<int> sils;

            overlap_cost();
            overlap_cost(std::vector<int> sils);

            virtual double operator()(std::vector<segment> const& gold_edges,
                segment const& e) const override;

        };

    }

}

#endif
