#ifndef FEAT_H
#define FEAT_H

#include "scrf/util.h"
#include "scrf/scrf.h"
#include "scrf/segfeat.h"

namespace scrf {

    struct composite_feature
        : public scrf_feature {

        std::vector<std::shared_ptr<scrf_feature>> features;

        composite_feature();
        composite_feature(std::string name);

        virtual void operator()(
            feat_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const override;

    private:
        std::string name_;
        
    };

    std::vector<double>& lexicalize(int order, feat_t& feat,
        fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e);

    struct segment_feature
        : public scrf_feature {

        segment_feature(
            int order,
            std::shared_ptr<segfeat::feature> raw_feat_func,
            std::vector<std::vector<real>> const& frames);

        int order;
        std::shared_ptr<segfeat::feature> feat_func;
        std::vector<std::vector<real>> const& frames;

        virtual void operator()(
            feat_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const override;
    };

    namespace feature {

        struct lm_score
            : public scrf_feature {

            virtual void operator()(
                feat_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct lattice_score
            : public scrf_feature {

            virtual void operator()(
                feat_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;

        };

        struct external_feature
            : public scrf_feature {

            int order;
            std::vector<int> dims;

            external_feature(int order, std::vector<int> dims);

            virtual void operator()(
                feat_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;

        };

        struct frame_feature
            : public scrf::scrf_feature {
        
            std::vector<std::vector<double>> const& frames;
            std::unordered_map<std::string, int> const& phone_id;
        
            frame_feature(std::vector<std::vector<double>> const& frames,
                std::unordered_map<std::string, int> const& phone_set);
        
            virtual void operator()(
                scrf::feat_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const;
        
        };

    }

}

#endif
