#ifndef SEG_COST_H
#define SEG_COST_H

#include "seg/seg-weight.h"
#include "seg/cost.h"

namespace seg {

    template <class fst_t>
    struct cost_t
        : public seg_weight<fst_t> {

        cost_t(std::shared_ptr<cost::cost_t<typename fst_t::output_symbol>> cost_func,
            std::vector<cost::segment<typename fst_t::output_symbol>> const& gold_segs);

        std::shared_ptr<cost::cost_t<typename fst_t::output_symbol>> cost_func;
        std::vector<cost::segment<typename fst_t::output_symbol>> const& gold_segs;

        virtual double operator()(fst_t const& f, typename fst_t::edge e) const override;

    };

    template <class fst_t>
    cost_t<fst_t> make_overlap_cost(
        std::vector<cost::segment<typename fst_t::output_symbol>> const& gold_segs,
        std::vector<typename fst_t::output_symbol> sils);

}

#include "seg/seg-cost-impl.h"

#endif
