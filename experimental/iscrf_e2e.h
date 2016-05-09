#ifndef ISCRF_E2E_H
#define ISCRF_E2E_H

#include "scrf/experimental/iscrf.h"

namespace iscrf {

    namespace e2e {

        struct nn_inference_args {
            lstm::dblstm_feat_param_t nn_param;
            rnn::pred_param_t pred_param;
            int subsample_freq;
            int subsample_shift;
            double rnndrop_prob;

            bool rnndrop;
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

            lstm::dblstm_feat_param_t nn_opt_data;
            rnn::pred_param_t pred_opt_data;
            int rnndrop_seed;
        };

        void parse_learning_args(learning_args& l_args,
            std::unordered_map<std::string, std::string> const& args);

        std::tuple<lstm::dblstm_feat_param_t, rnn::pred_param_t>
        load_lstm_param(std::string filename);

        void save_lstm_param(lstm::dblstm_feat_param_t const& nn_param,
            rnn::pred_param_t const& pred_param,
            std::string filename);

        std::shared_ptr<scrf::composite_feature_with_frame_grad<ilat::fst, scrf::dense_vec>>
        filter_feat_with_frame_grad(iscrf_data const& data);

        std::vector<std::shared_ptr<autodiff::op_t>>
        make_input(autodiff::computation_graph& comp_graph,
            lstm::dblstm_feat_nn_t& nn,
            rnn::pred_nn_t& pred_nn,
            std::vector<std::vector<double>> const& frames,
            std::default_random_engine& gen,
            nn_inference_args& nn_args);
    }

}

#endif
