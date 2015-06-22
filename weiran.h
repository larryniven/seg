#ifndef WEIRAN_H
#define WEIRAN_H

#include "scrf/util.h"
#include "scrf/scrf.h"
#include "scrf/nn.h"
#include <vector>

namespace weiran {

    nn::nn_t make_nn(nn::param_t const& param);

    struct weiran_feature
        : public scrf::scrf_feature {

        std::vector<std::vector<real>> const& frames;
        std::vector<real> const& cm_mean;
        std::vector<real> const& cm_stddev;
        nn::nn_t nn;
        int start_dim;
        int end_dim;

        mutable std::unordered_map<int, std::vector<real>> feat_cache;

        weiran_feature(
            std::vector<std::vector<real>> const& frames,
            std::vector<real> const& cm_mean,
            std::vector<real> const& cm_stddev,
            nn::nn_t nn,
            int start_dim = -1,
            int end_dim = -1);

        virtual int size() const override;

        virtual std::string name() const override;

        virtual void operator()(
            scrf::param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const override;

    };

}

#endif
