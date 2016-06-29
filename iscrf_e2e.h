#ifndef ISCRF_E2E_H
#define ISCRF_E2E_H

#include "scrf/iscrf.h"
#include "nn/tensor_tree.h"

namespace iscrf {

    namespace e2e {

        struct nn_inference_args {
            std::shared_ptr<tensor_tree::vertex> nn_param;
            std::shared_ptr<tensor_tree::vertex> pred_param;
            int layer;
            double dropout;
            bool frame_softmax;
        };

        struct inference_args
            : public ::iscrf::inference_args
            , public nn_inference_args {

        };

        void parse_nn_inference_args(nn_inference_args& i_args,
            std::unordered_map<std::string, std::string> const& args);

        void parse_inference_args(inference_args& i_args,
            std::unordered_map<std::string, std::string> const& args);

        struct learning_args
            : public ::iscrf::learning_args
            , public nn_inference_args {

            std::shared_ptr<tensor_tree::vertex> nn_opt_data;
            std::shared_ptr<tensor_tree::vertex> pred_opt_data;
            double clip;
            int dropout_seed;
        };

        void parse_learning_args(learning_args& l_args,
            std::unordered_map<std::string, std::string> const& args);

        std::tuple<int, std::shared_ptr<tensor_tree::vertex>, std::shared_ptr<tensor_tree::vertex>>
        load_lstm_param(std::string filename);

        void save_lstm_param(std::shared_ptr<tensor_tree::vertex> nn_param,
            std::shared_ptr<tensor_tree::vertex> pred_param,
            std::string filename);

        std::shared_ptr<scrf::composite_feature_with_frame_grad<ilat::fst, scrf::dense_vec>>
        filter_feat_with_frame_grad(iscrf_data const& data);

        std::vector<std::shared_ptr<autodiff::op_t>>
        make_feat(autodiff::computation_graph& comp_graph,
            std::shared_ptr<tensor_tree::vertex> lstm_tree_var,
            std::shared_ptr<tensor_tree::vertex> pred_tree_var,
            lstm::stacked_bi_lstm_nn_t& nn,
            rnn::pred_nn_t& pred_nn,
            std::vector<std::vector<double>> const& frames,
            std::default_random_engine& gen,
            nn_inference_args& nn_args);

    }

}

#endif
