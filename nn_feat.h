#ifndef NN_FEAT_H
#define NN_FEAT_H

#include "scrf/scrf_feat.h"
#include "la/la.h"
#include "autodiff/autodiff.h"

namespace scrf {

    namespace nn {

        struct param_t {
            std::vector<la::matrix<double>> weight;
            std::vector<la::vector<double>> bias;
        };

        void iadd(param_t& p1, param_t const& p2);
        void isub(param_t& p1, param_t const& p2);
        void imul(param_t& p, double c);

        param_t load_param(std::istream& is);
        param_t load_param(std::string filename);

        void save_param(param_t const& p, std::ostream& os);
        void save_param(param_t const& p, std::string filename);

        void adagrad_update(param_t& param, param_t const& grad,
            param_t& accu_grad_sq, double step_size);

        struct nn_t {
            std::vector<std::shared_ptr<autodiff::op_t>> hidden;

            std::vector<std::shared_ptr<autodiff::op_t>> weight;
            std::vector<std::shared_ptr<autodiff::op_t>> bias;
        };

        nn_t make_nn(param_t const& p);

        param_t copy_grad(nn_t const& nn);

        struct nn_feature
            : public scrf_feature {

            std::shared_ptr<scrf_feature> base_feat;
            nn_t nn;

            std::unordered_map<std::string, std::vector<double>> label_vec;

            nn_feature(std::unordered_map<std::string, int> const& label_id);

            virtual void operator()(
                feat_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;

            param_t grad(
                scrf::param_t const& param,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const;

        };

        param_t hinge_grad(fst::path<scrf::scrf_t> const& gold_path,
            fst::path<scrf::scrf_t> const& graph_path,
            nn_feature const& nn_feat,
            scrf::param_t const& param);

    }

}

#endif
