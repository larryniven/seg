#include "seg/iscrf_e2e_ff.h"
#include <fstream>

namespace iscrf {

    namespace e2e_ff {

        void parse_nn_inference_args(nn_inference_args& nn_args,
            std::unordered_map<std::string, std::string> const& args)
        {
            nn_args.nn_param = residual::load_nn_param(args.at("nn-param"));
        }

        void parse_inference_args(inference_args& i_args,
            std::unordered_map<std::string, std::string> const& args)
        {
            ::iscrf::parse_inference_args(i_args, args);
            parse_nn_inference_args(i_args, args);
        }

        void parse_learning_args(learning_args& l_args,
            std::unordered_map<std::string, std::string> const& args)
        {
            ::iscrf::parse_learning_args(l_args, args);
            parse_nn_inference_args(l_args, args);

            l_args.nn_opt_data = residual::load_nn_param(args.at("nn-opt-data"));
        }

        std::vector<residual::nn_t> make_nn(
            autodiff::computation_graph& comp_graph,
            std::vector<std::vector<double>> const& frames,
            residual::nn_param_t& nn_param)
        {
            std::vector<residual::nn_t> result;

            int dim = frames.front().size();

            for (int i = 0; i < frames.size(); ++i) {
                la::vector<double> v;
                v.resize(dim * 11);

                for (int k = -5; k <= 5; k++) {
                    if (0 <= i + k && i + k < frames.size()) {
                        for (int j = 0; j < dim; ++j) {
                            v((k + 5) * dim + j) = frames[i + k][j];
                        }
                    }
                }

                result.push_back(residual::make_nn(comp_graph, nn_param));
                result.back().input->output = std::make_shared<la::vector<double>>(std::move(v));
            }

            return result;
        }

        std::vector<std::vector<double>> make_input(std::vector<residual::nn_t> const& nns)
        {
            std::vector<std::vector<double>> result;

            for (auto& nn: nns) {
                autodiff::eval(nn.layer.back().output, autodiff::eval_funcs);
                auto& v = autodiff::get_output<la::vector<double>>(nn.layer.back().output);
                result.push_back(std::vector<double> { v.data(), v.data() + v.size() });
            }

            return result;
        }

    }

}
