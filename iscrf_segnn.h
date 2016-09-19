#ifndef ISCRF_SEGNN_H
#define ISCRF_SEGNN_H

#include "seg/segnn.h"
#include "seg/iscrf.h"

namespace iscrf {

    namespace segnn {

        struct nn_inference_args {
            ::segnn::param_t nn_param;
            ::segnn::nn_t nn;
        };

        void parse_nn_inference_args(nn_inference_args& i_args,
            std::unordered_map<std::string, std::string> const& args);

        struct inference_args
            : public ::iscrf::inference_args
            , public nn_inference_args {
        };

        void parse_inference_args(inference_args& i_args,
            std::unordered_map<std::string, std::string> const& args);

        struct learning_args
            : public ::iscrf::learning_args
            , public nn_inference_args {

            ::segnn::param_t nn_opt_data;
        };

        void parse_learning_args(learning_args& l_args,
            std::unordered_map<std::string, std::string> const& args);

        ::segnn::segnn_feat make_segnn_feat(
            std::vector<std::string> features,
            std::vector<std::vector<double>> const& frames,
            nn_inference_args const& i_args,
            std::unordered_map<std::string, std::string> const& args);

        void parameterize(iscrf_data& data,
            ::segnn::segnn_feat const& segnn_feat,
            ::iscrf::inference_args const& i_args);

        void parameterize(learning_sample& s, learning_args const& l_args);
    }

}

#endif
