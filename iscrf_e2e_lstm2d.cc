#include "scrf/iscrf_e2e_lstm2d.h"
#include <fstream>

namespace iscrf {

    namespace e2e_lstm2d {

        void parse_nn_inference_args(nn_inference_args& nn_args,
            std::unordered_map<std::string, std::string> const& args)
        {
            nn_args.nn_param = lstm::load_db_lstm2d_param(args.at("nn-param"));
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

            l_args.nn_opt_data = lstm::load_db_lstm2d_param(args.at("nn-opt-data"));
        }

    }

}
