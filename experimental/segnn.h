#ifndef SEGNN_H
#define SEGNN_H

#include "scrf/experimental/segfeat.h"
#include "autodiff/autodiff.h"
#include "scrf/experimental/scrf.h"
#include "scrf/experimental/scrf_feat.h"

namespace segnn {

    struct param_t {
        la::matrix<double> label_embedding;
        la::matrix<double> duration_embedding;

        la::matrix<double> feat_weight;
        la::matrix<double> label_weight;
        la::matrix<double> duration_weight;
        la::vector<double> bias;

        std::vector<la::matrix<double>> layer_weight;
        std::vector<la::vector<double>> layer_bias;
    };

    void resize_as(param_t& p1, param_t const& p2);
    void iadd(param_t& p1, param_t const& p2);

    param_t load_nn_param(std::istream& is);
    param_t load_nn_param(std::string filename);
    void save_nn_param(param_t const& param, std::ostream& os);
    void save_nn_param(param_t const& param, std::string filename);

    void rmsprop_update(param_t& param, param_t const& grad,
        param_t& opt_data, double decay, double step_size);
    void adagrad_update(param_t& param, param_t const& grad,
        param_t& accu_grad_sq, double step_size);

    struct nn_t {
        std::shared_ptr<autodiff::op_t> feat_weight;
        std::shared_ptr<autodiff::op_t> label_weight;
        std::shared_ptr<autodiff::op_t> duration_weight;
        std::shared_ptr<autodiff::op_t> bias;

        std::shared_ptr<autodiff::op_t> feat_embedding;
        std::shared_ptr<autodiff::op_t> label_embedding;
        std::shared_ptr<autodiff::op_t> duration_embedding;

        std::vector<std::shared_ptr<autodiff::op_t>> layer_weight;
        std::vector<std::shared_ptr<autodiff::op_t>> layer_bias;

        std::vector<std::shared_ptr<autodiff::op_t>> layer;

        std::shared_ptr<autodiff::op_t> output;
    };

    nn_t make_nn(autodiff::computation_graph& graph,
        param_t const& param);

    param_t copy_nn_grad(nn_t const& nn);

    struct segnn_feat
        : public scrf::scrf_feature_with_grad<ilat::fst, scrf::dense_vec> {

        segnn_feat(
            std::vector<std::vector<double>> const& frames,
            std::shared_ptr<segfeat::feature> const& base_feat,
            param_t const& param, nn_t const& nn);

        std::vector<std::vector<double>> const& frames;
        std::shared_ptr<segfeat::feature> base_feat;

        param_t const& param;
        nn_t const& nn;

        param_t gradient;
        param_t zero;

        virtual void operator()(scrf::dense_vec& f, ilat::fst const& a,
            int e) const override;

        virtual void grad(scrf::dense_vec const& g, ilat::fst const& a,
            int e) override;

    };

}

#endif
