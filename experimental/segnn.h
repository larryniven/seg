#ifndef SEGNN_H
#define SEGNN_H

#include "scrf/experimental/segfeat.h"
#include "autodiff/autodiff.h"
#include "nn/residual.h"
#include "scrf/experimental/scrf.h"
#include "scrf/experimental/scrf_feat.h"

namespace segnn {

    struct segnn_feat
        : public scrf::scrf_feature_with_grad<ilat::fst, scrf::dense_vec> {

        segnn_feat(
            scrf::feat_dim_alloc& alloc,
            std::vector<std::vector<double>> const& frames,
            std::shared_ptr<segfeat::feature> const& base_feat,
            residual::nn_param_t const& param, residual::nn_t const& nn);

        scrf::feat_dim_alloc& alloc;
        int dim;
        std::vector<std::vector<double>> const& frames;
        std::shared_ptr<segfeat::feature> base_feat;
        residual::nn_param_t const& param;
        residual::nn_t const& nn;
        residual::nn_param_t gradient;
        residual::nn_param_t zero;

        virtual void operator()(scrf::dense_vec& f, ilat::fst const& a,
            int e) const override;

        virtual void grad(scrf::dense_vec const& g, ilat::fst const& a,
            int e) override;

    };

}

#endif
