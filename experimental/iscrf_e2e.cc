#include "scrf/experimental/iscrf_e2e.h"
#include <fstream>

namespace iscrf {

    namespace e2e {

        void parse_nn_inference_args(nn_inference_args& i_args,
            std::unordered_map<std::string, std::string> const& args)
        {
            std::tie(i_args.nn_param, i_args.pred_param) = load_lstm_param(args.at("nn-param"));

            i_args.rnndrop_prob = 1;
            if (ebt::in(std::string("rnndrop-prob"), args)) {
                i_args.rnndrop_prob = std::stod(args.at("rnndrop-prob"));
                assert(0 <= i_args.rnndrop_prob && i_args.rnndrop_prob <= 1);
            }

            i_args.subsample_freq = 1;
            if (ebt::in(std::string("subsample-freq"), args)) {
                i_args.subsample_freq = std::stoi(args.at("subsample-freq"));
            }

            i_args.subsample_shift = 0;
            if (ebt::in(std::string("subsample-shift"), args)) {
                i_args.subsample_shift = std::stoi(args.at("subsample-shift"));
            }
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

            std::tie(l_args.nn_opt_data, l_args.pred_opt_data)
                = load_lstm_param(args.at("nn-opt-data"));

            l_args.momentum = -1;
            if (ebt::in(std::string("momentum"), args)) {
                l_args.momentum = std::stod(args.at("momentum"));
                assert(0 <= l_args.momentum && l_args.momentum <= 1);
            }

            l_args.decay = -1;
            if (ebt::in(std::string("decay"), args)) {
                l_args.decay = std::stod(args.at("decay"));
                assert(0 <= l_args.decay && l_args.decay <= 1);
            }

            l_args.rnndrop_seed = 0;
            if (ebt::in(std::string("rnndrop-seed"), args)) {
                l_args.rnndrop_seed = std::stoi(args.at("rnndrop-seed"));
            }

            l_args.sils.push_back(l_args.label_id.at("sil"));
        }

        std::tuple<lstm::dblstm_feat_param_t, rnn::pred_param_t>
        load_lstm_param(std::string filename)
        {
            std::ifstream ifs { filename };

            lstm::dblstm_feat_param_t nn_param = lstm::load_dblstm_feat_param(ifs);
            rnn::pred_param_t pred_param = rnn::load_pred_param(ifs);

            return std::make_tuple(nn_param, pred_param);
        }

        void save_lstm_param(lstm::dblstm_feat_param_t const& nn_param,
            rnn::pred_param_t const& pred_param,
            std::string filename)
        {
            std::ofstream ofs { filename };

            lstm::save_dblstm_feat_param(nn_param, ofs);
            rnn::save_pred_param(pred_param, ofs);
        }

        std::shared_ptr<scrf::composite_feature_with_frame_grad<ilat::fst, scrf::dense_vec>>
        filter_feat_with_frame_grad(iscrf_data const& data)
        {
            scrf::composite_feature_with_frame_grad<ilat::fst, scrf::dense_vec> result;

            for (int i = 0; i < data.features->size(); ++i) {
                auto& k = data.features->at(i);

                if (ebt::startswith(k, "frame-samples")
                        || ebt::startswith(k, "frame-avg")
                        || ebt::startswith(k, "right-boundary")
                        || ebt::startswith(k, "left-boundary")) {
                    result.features.push_back(std::dynamic_pointer_cast<
                        scrf::scrf_feature_with_frame_grad<ilat::fst, scrf::dense_vec>>(
                        data.feature_func->features[i]));
                }
            }

            return std::make_shared<scrf::composite_feature_with_frame_grad<ilat::fst,
                scrf::dense_vec>>(result);
        }

    }

}
