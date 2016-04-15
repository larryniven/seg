#ifndef SCRF_COST_H
#define SCRF_COST_H

#include "scrf/experimental/scrf.h"
#include "scrf/experimental/segcost.h"

namespace scrf {

    template <class fst>
    struct seg_cost
        : public scrf_weight<fst> {

        seg_cost(std::shared_ptr<segcost::cost<typename fst::symbol>> cost,
            fst const& gold_path);

        std::shared_ptr<segcost::cost<typename fst::symbol>> cost;
        std::vector<segcost::segment<typename fst::symbol>> gold_segs;

        virtual double operator()(fst const& f, typename fst::edge e) const override;

    };

    template <class fst>
    seg_cost<fst>::seg_cost(std::shared_ptr<segcost::cost<typename fst::symbol>> cost,
        fst const& gold_path)
        : cost(cost)
    {
        for (auto& e: gold_path.edges()) {
            auto tail = gold_path.tail(e);
            auto head = gold_path.head(e);

            gold_segs.push_back(segcost::segment<typename fst::symbol> {
                gold_path.time(tail), gold_path.time(head), gold_path.output(e) });
        }
    }

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
    seg_cost<fst> make_overlap_cost(fst const& gold,
        std::vector<typename fst::symbol> sils)
    {
        return seg_cost<fst> { std::make_shared<segcost::overlap_cost<typename fst::symbol>>(
            segcost::overlap_cost<typename fst::symbol>{ sils }), gold };
    }

}

#endif
