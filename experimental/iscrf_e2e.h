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
        };

        struct inference_args
            : public ::iscrf::inference_args
            , public nn_inference_args {

        };

        void parse_inference_args(inference_args& i_args,
            std::unordered_map<std::string, std::string> const& args);

        struct learning_args
            : public ::iscrf::learning_args
            , public nn_inference_args {

            scrf::dense_vec opt_data;
            lstm::dblstm_feat_param_t nn_opt_data;
            rnn::pred_param_t pred_opt_data;
            double step_size;
            double momentum;
            double decay;
            std::vector<int> sils;
            double cost_scale;
            int rnndrop_seed;
        };

        void parse_learning_args(learning_args& l_args,
            std::unordered_map<std::string, std::string> const& args);

        std::tuple<lstm::dblstm_feat_param_t, rnn::pred_param_t>
        load_lstm_param(std::string filename);

        void save_lstm_param(lstm::dblstm_feat_param_t const& nn_param,
            rnn::pred_param_t const& pred_param,
            std::string filename);

        std::shared_ptr<scrf::scrf_feature_with_frame_grad<ilat::fst, scrf::dense_vec>>
        filter_feat_with_frame_grad(iscrf_data const& data);

    }

}

#endif
