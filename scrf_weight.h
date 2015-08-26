#ifndef SCRF_WEIGHT_H
#define SCRF_WEIGHT_H

#include "scrf.h"

namespace scrf {

    struct composite_weight
        : public scrf_weight {

        std::vector<std::shared_ptr<scrf_weight>> weights;

        virtual real operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const override;

    };

    std::shared_ptr<scrf_weight> operator+(std::shared_ptr<scrf_weight> w1,
        std::shared_ptr<scrf_weight> w2);

    composite_weight make_weight(
        param_t const& param,
        composite_feature const& feat);

}


#endif
