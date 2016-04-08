#ifndef SCRF_COST_H
#define SCRF_COST_H

#include "scrf/scrf.h"
#include "scrf/segcost.h"

namespace scrf {

    struct seg_cost
        : public scrf_weight {

        seg_cost(std::shared_ptr<segcost::cost> cost,
            fst::path<scrf_t> const& gold_fst);

        std::shared_ptr<segcost::cost> cost;
        std::vector<speech::segment> gold_segs;

        virtual real operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const override;

    };

    seg_cost make_overlap_cost(fst::path<scrf_t> const& gold_fst);

    struct backoff_cost
        : public scrf_weight {

        virtual real operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const;
    };

    struct overlap_cost
        : public scrf_weight {

        fst::path<scrf_t> const& gold;

        mutable std::unordered_map<int, std::vector<std::tuple<int, int>>> edge_cache;

        overlap_cost(fst::path<scrf_t> const& gold);

        virtual real operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const;

    };

    struct neg_cost
        : public scrf_weight {

        std::shared_ptr<scrf_weight> cost;

        neg_cost(std::shared_ptr<scrf_weight> cost);

        virtual real operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const;
    };

    namespace first_order {

        struct seg_cost
            : public scrf_weight {

            seg_cost(std::shared_ptr<segcost::first_order::cost> cost,
                fst::path<scrf_t> const& gold_fst);

            std::shared_ptr<segcost::first_order::cost> cost;
            std::vector<segcost::first_order::segment> gold_segs;

            virtual real operator()(ilat::fst const& fst,
                int e) const override;

        };

        struct cached_seg_cost
            : public scrf_weight {

            cached_seg_cost(std::shared_ptr<segcost::first_order::cost> cost,
                fst::path<scrf_t> const& gold_fst);

            std::shared_ptr<segcost::first_order::cost> cost;
            std::vector<segcost::first_order::segment> gold_segs;

            mutable std::vector<double> cache;
            mutable std::vector<bool> in_cache;

            virtual real operator()(ilat::fst const& fst,
                int e) const override;

        };

        seg_cost make_overlap_cost(fst::path<scrf_t> const& gold_fst, std::vector<int> sils);
        cached_seg_cost make_cached_overlap_cost(fst::path<scrf_t> const& gold_fst, std::vector<int> sils);

        struct neg_cost
            : public scrf_weight {

            std::shared_ptr<scrf_weight> cost;

            neg_cost(std::shared_ptr<scrf_weight> cost);

            virtual real operator()(ilat::fst const& fst,
                int e) const;
        };

    }

    namespace experimental {

        template <class fst>
        struct seg_cost
            : public scrf_weight<fst> {

            seg_cost(std::shared_ptr<segcost::experimental::cost<typename fst::symbol>> cost,
                fst const& gold_path);

            std::shared_ptr<segcost::experimental::cost<typename fst::symbol>> cost;
            std::vector<segcost::experimental::segment<typename fst::symbol>> gold_segs;

            virtual double operator()(fst const& f, typename fst::edge e) const override;

        };

        template <class fst>
        seg_cost<fst>::seg_cost(std::shared_ptr<segcost::experimental::cost<typename fst::symbol>> cost,
            fst const& gold_path)
            : cost(cost)
        {
            for (auto& e: gold_path.edges()) {
                int tail = gold_path.tail(e);
                int head = gold_path.head(e);

                gold_segs.push_back(segcost::experimental::segment<typename fst::symbol> {
                    gold_path.time(tail), gold_path.time(head), gold_path.output(e) });
            }
        }

        template <class fst>
        double seg_cost<fst>::operator()(fst const& f,
            typename fst::edge e) const
        {
            int tail = f.tail(e);
            int head = f.head(e);

            return (*cost)(gold_segs, segcost::experimental::segment<typename fst::symbol> {
                f.time(tail), f.time(head), f.output(e) });
        }

        template <class fst>
        seg_cost<fst> make_overlap_cost(fst const& gold,
            std::vector<typename fst::symbol> sils)
        {
            return seg_cost<fst> { std::make_shared<segcost::experimental::overlap_cost<typename fst::symbol>>(
                segcost::experimental::overlap_cost<typename fst::symbol>{ sils }), gold };
        }

    }

}

#endif
