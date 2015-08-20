#ifndef MAKE_FEATURE_H
#define MAKE_FEATURE_H

#include "scrf/feat.h"
#include "scrf/scrf.h"
#include "scrf/nn.h"
#include "scrf/weiran.h"

namespace scrf {

    struct lexicalized_feature
        : public scrf_feature {

        lexicalized_feature(
            int order,
            std::shared_ptr<segfeat::feature> raw_feat_func,
            std::vector<std::vector<real>> const& frames);

        int order;
        std::shared_ptr<segfeat::feature> feat_func;
        std::vector<std::vector<real>> const& frames;

        virtual int size() const override;
        virtual std::string name() const override;

        virtual void operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const override;
    };

    composite_feature make_feature(
        std::vector<std::string> features,
        std::vector<std::vector<real>> const& inputs, int max_seg);

    composite_feature make_feature(
        std::vector<std::string> features,
        std::vector<std::vector<real>> const& inputs, int max_seg,
        std::vector<real> const& cm_mean, std::vector<real> const& cm_stddev,
        nn::nn_t const& nn);

    composite_weight make_weight(
        param_t const& param,
        composite_feature const& feat);

    composite_feature make_feature2(
        std::vector<std::string> features,
        std::vector<std::vector<real>> const& frames);

    composite_feature make_feature3(
        std::vector<std::string> features);

}

#endif
