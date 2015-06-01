#ifndef WEIRAN_H
#define WEIRAN_H

#include "scrf/util.h"
#include "scrf/scrf.h"
#include "autodiff/autodiff.h"
#include <vector>

namespace weiran {

    struct param_t {
        std::vector<std::vector<std::vector<real>>> weight;
    };

    struct nn_t {
        std::vector<std::shared_ptr<autodiff::op>> layers;
    };

    param_t load_param(std::istream& is);
    param_t load_param(std::string filename);

    void save_param(param_t const& param, std::ostream& os);
    void save_param(param_t const& param, std::string filename);

    nn_t make_nn(param_t const& param);

    void move_in_param(nn_t& nn, param_t& param);

    void move_out_param(nn_t& nn, param_t& param);

    struct weiran_feature
        : public scrf::scrf_feature {

        std::vector<std::vector<real>> const& frames;
        std::vector<real> const& cm_mean;
        std::vector<real> const& cm_stddev;
        nn_t nn;
        int start_dim;
        int end_dim;

        mutable std::unordered_map<int, std::vector<real>> feat_cache;

        weiran_feature(
            std::vector<std::vector<real>> const& frames,
            std::vector<real> const& cm_mean,
            std::vector<real> const& cm_stddev,
            nn_t nn,
            int start_dim = -1,
            int end_dim = -1);

        virtual void operator()(
            scrf::param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const override;
    };

}

#endif
