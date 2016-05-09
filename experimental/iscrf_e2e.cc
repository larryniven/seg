#include "scrf/experimental/iscrf_e2e.h"
#include <fstream>

namespace iscrf {

    namespace e2e {

        void parse_nn_inference_args(nn_inference_args& nn_args,
            std::unordered_map<std::string, std::string> const& args)
        {
            std::tie(nn_args.nn_param, nn_args.pred_param) = load_lstm_param(args.at("nn-param"));

            nn_args.rnndrop_prob = 1;
            nn_args.rnndrop = false;
            if (ebt::in(std::string("rnndrop-prob"), args)) {
                nn_args.rnndrop_prob = std::stod(args.at("rnndrop-prob"));
                assert(0 <= nn_args.rnndrop_prob && nn_args.rnndrop_prob <= 1);

                nn_args.rnndrop = true;
            }

            nn_args.subsample_freq = 1;
            if (ebt::in(std::string("subsample-freq"), args)) {
                nn_args.subsample_freq = std::stoi(args.at("subsample-freq"));
            }

            nn_args.subsample_shift = 0;
            if (ebt::in(std::string("subsample-shift"), args)) {
                nn_args.subsample_shift = std::stoi(args.at("subsample-shift"));
            }

            nn_args.frame_softmax = false;
            if (ebt::in(std::string("frame-softmax"), args)) {
                nn_args.frame_softmax = true;
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

        std::vector<std::shared_ptr<autodiff::op_t>>
        make_input(autodiff::computation_graph& comp_graph,
            lstm::dblstm_feat_nn_t& nn,
            rnn::pred_nn_t& pred_nn,
            std::vector<std::vector<double>> const& frames,
            std::default_random_engine& gen,
            nn_inference_args& nn_args)
        {
            std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
            for (auto& f: frames) {
                frame_ops.push_back(comp_graph.var(la::vector<double>(f)));
            }

            std::vector<std::shared_ptr<autodiff::op_t>> subsampled_input;
            if (nn_args.subsample_freq > 1) {
                subsampled_input = rnn::subsample_input(frame_ops,
                    nn_args.subsample_freq, nn_args.subsample_shift);
            } else {
                subsampled_input = frame_ops;
            }

            nn = lstm::make_dblstm_feat_nn(
                comp_graph, nn_args.nn_param, subsampled_input);

            if (nn_args.frame_softmax) {
                lstm::apply_random_mask(nn, nn_args.nn_param, gen, nn_args.rnndrop_prob);
            }

            std::vector<std::shared_ptr<autodiff::op_t>> output;

            if (nn_args.rnndrop) {
                pred_nn = rnn::make_pred_nn(comp_graph,
                    nn_args.pred_param, nn.layer.back().output);

                output = pred_nn.logprob;
            } else {
                output = nn.layer.back().output;
            }

            std::vector<std::shared_ptr<autodiff::op_t>> upsampled_output;
            if (nn_args.subsample_freq > 1) {
                 upsampled_output = rnn::upsample_output(output,
                     nn_args.subsample_freq, nn_args.subsample_shift, frames.size());
            } else {
                 upsampled_output = output;
            }

            return upsampled_output;
        }
    }

}
