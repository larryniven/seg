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

}

#endif
