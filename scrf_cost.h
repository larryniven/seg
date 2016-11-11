#ifndef SCRF_COST_H
#define SCRF_COST_H

#include "seg/scrf.h"
#include "seg/segcost.h"

namespace scrf {

    template <class fst>
    struct seg_cost
        : public scrf_weight<fst> {

        seg_cost(std::shared_ptr<segcost::cost<typename fst::symbol>> cost,
            std::vector<segcost::segment<typename fst::symbol>> const& gold_segs);

        std::shared_ptr<segcost::cost<typename fst::symbol>> cost;
        std::vector<segcost::segment<typename fst::symbol>> const& gold_segs;

        virtual double operator()(fst const& f, typename fst::edge e) const override;

    };

    template <class fst>
    seg_cost<fst>::seg_cost(std::shared_ptr<segcost::cost<typename fst::symbol>> cost,
        std::vector<segcost::segment<typename fst::symbol>> const& gold_segs)
        : cost(cost), gold_segs(gold_segs)
    {}

    template <class fst>
    double seg_cost<fst>::operator()(fst const& f,
        typename fst::edge e) const
    {
        auto tail = f.tail(e);
        auto head = f.head(e);

        return (*cost)(gold_segs, segcost::segment<typename fst::symbol> {
            f.time(tail), f.time(head), f.output(e) });
    }

    template <class fst>
    seg_cost<fst> make_hit_cost(
        std::vector<segcost::segment<typename fst::symbol>> const& gold_segs)
    {
        return seg_cost<fst> { std::make_shared<segcost::hit_cost<typename fst::symbol>>(
            segcost::hit_cost<typename fst::symbol>{}), gold_segs };
    }

    template <class fst>
    seg_cost<fst> make_overlap_cost(
        std::vector<segcost::segment<typename fst::symbol>> const& gold_segs,
        std::vector<typename fst::symbol> sils)
    {
        return seg_cost<fst> { std::make_shared<segcost::overlap_cost<typename fst::symbol>>(
            segcost::overlap_cost<typename fst::symbol>{ sils }), gold_segs };
    }

    template <class fst>
    seg_cost<fst> make_overlap_portion_cost(
        std::vector<segcost::segment<typename fst::symbol>> const& gold_segs,
        std::vector<typename fst::symbol> sils)
    {
        return seg_cost<fst> { std::make_shared<segcost::overlap_portion_cost<typename fst::symbol>>(
            segcost::overlap_portion_cost<typename fst::symbol>{ sils }), gold_segs };
    }

    template <class fst>
    seg_cost<fst> make_cover_cost(
        std::vector<segcost::segment<typename fst::symbol>> const& gold_segs,
        std::vector<typename fst::symbol> sils)
    {
        return seg_cost<fst> { std::make_shared<segcost::cover_cost<typename fst::symbol>>(
            segcost::cover_cost<typename fst::symbol>{ sils }), gold_segs };
    }

}

#endif
