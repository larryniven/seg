#ifndef ISCRF_E2E_LSTM2D_H
#define ISCRF_E2E_LSTM2D_H

#include "scrf/iscrf.h"
#include "nn/lstm.h"

namespace iscrf {

    namespace e2e_lstm2d {

        struct nn_inference_args {
            lstm::db_lstm2d_param_t nn_param;
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

            lstm::db_lstm2d_param_t nn_opt_data;
        };

        void parse_learning_args(learning_args& l_args,
            std::unordered_map<std::string, std::string> const& args);

    }

}

#endif
