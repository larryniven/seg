#ifndef ISCRF_E2E_FF_H
#define ISCRF_E2E_FF_H

#include "scrf/experimental/iscrf.h"
#include "nn/residual.h"

namespace iscrf {

    namespace e2e_ff {

        struct nn_inference_args {
            residual::nn_param_t nn_param;
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

            residual::nn_param_t nn_opt_data;
        };

        void parse_learning_args(learning_args& l_args,
            std::unordered_map<std::string, std::string> const& args);

        std::vector<residual::nn_t> make_nn(
            autodiff::computation_graph& comp_graph,
            std::vector<std::vector<double>> const& frames,
            residual::nn_param_t& nn_param);

        std::vector<std::vector<double>> make_input(std::vector<residual::nn_t> const& nns);
    }

}

#endif
