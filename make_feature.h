#ifndef MAKE_FEATURE_H
#define MAKE_FEATURE_H

#include "scrf/scrf.h"
#include "scrf/nn.h"
#include "scrf/weiran.h"

namespace scrf {

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

}

#endif
