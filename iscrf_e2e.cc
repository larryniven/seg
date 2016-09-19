#include "seg/iscrf_e2e.h"
#include <fstream>

namespace iscrf {

    namespace e2e {

        void parse_nn_inference_args(nn_inference_args& nn_args,
            std::unordered_map<std::string, std::string> const& args)
        {
            std::tie(nn_args.layer, nn_args.nn_param, nn_args.pred_param)
                = load_lstm_param(args.at("nn-param"));

            nn_args.dropout = 0;
            if (ebt::in(std::string("dropout"), args)) {
                nn_args.dropout = std::stod(args.at("dropout"));
                assert(0 <= nn_args.dropout && nn_args.dropout <= 1);
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

            std::tie(l_args.layer, l_args.nn_opt_data, l_args.pred_opt_data)
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

            l_args.dropout_seed = 0;
            if (ebt::in(std::string("dropout-seed"), args)) {
                l_args.dropout_seed = std::stoi(args.at("dropout-seed"));
            }

            l_args.clip = 0;
            if (ebt::in(std::string("clip"), args)) {
                l_args.clip = std::stod(args.at("clip"));
            }
        }

        std::tuple<int, std::shared_ptr<tensor_tree::vertex>, std::shared_ptr<tensor_tree::vertex>>
        load_lstm_param(std::string filename)
        {
            std::ifstream ifs { filename };
            std::string line;

            std::getline(ifs, line);
            int layer = std::stoi(line);

            std::shared_ptr<tensor_tree::vertex> nn_param
                = lstm::make_stacked_bi_lstm_tensor_tree(layer);
            tensor_tree::load_tensor(nn_param, ifs);
            std::shared_ptr<tensor_tree::vertex> pred_param = nn::make_pred_tensor_tree();
            tensor_tree::load_tensor(pred_param, ifs);

            return std::make_tuple(layer, nn_param, pred_param);
        }

        void save_lstm_param(std::shared_ptr<tensor_tree::vertex> nn_param,
            std::shared_ptr<tensor_tree::vertex> pred_param,
            std::string filename)
        {
            std::ofstream ofs { filename };

            ofs << nn_param->children.size() << std::endl;
            tensor_tree::save_tensor(nn_param, ofs);
            tensor_tree::save_tensor(pred_param, ofs);
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
        make_feat(autodiff::computation_graph& comp_graph,
            std::shared_ptr<tensor_tree::vertex> lstm_var_tree,
            std::shared_ptr<tensor_tree::vertex> pred_var_tree,
            lstm::stacked_bi_lstm_nn_t& nn,
            rnn::pred_nn_t& pred_nn,
            std::vector<std::vector<double>> const& frames,
            std::default_random_engine& gen,
            nn_inference_args& nn_args)
        {
            std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
            for (auto& f: frames) {
                frame_ops.push_back(comp_graph.var(la::vector<double>(f)));
            }

            if (nn_args.dropout == 0) {
                nn = lstm::make_stacked_bi_lstm_nn(lstm_var_tree, frame_ops, lstm::lstm_builder{});
            } else {
                nn = lstm::make_stacked_bi_lstm_nn_with_dropout(comp_graph, lstm_var_tree,
                    frame_ops, lstm::lstm_builder{}, gen, nn_args.dropout);
            }

            std::vector<std::shared_ptr<autodiff::op_t>> output;

            if (nn_args.frame_softmax) {
                pred_nn = rnn::make_pred_nn(pred_var_tree, nn.layer.back().output);

                return pred_nn.logprob;
            } else {
                return nn.layer.back().output;
            }
        }
    }

}
