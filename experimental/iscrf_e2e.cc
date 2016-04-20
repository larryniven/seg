#include "iscrf_e2e.h"
#include <fstream>

namespace scrf {

    namespace e2e {

        void iscrf::feature(dense_vec& f, int e) const
        {
            (*feature_func)(f, *fst, e);
        }

        void iscrf::frame_grad(std::vector<std::vector<double>>& f,
            dense_vec const& param, int e) const
        {
            feature_func_with_frame_grad->frame_grad(f, param, *fst, e);
        }

        std::shared_ptr<iscrf> iscrf_path_maker::operator()(std::vector<int> const& edges,
            iscrf const& f) const
        {
            iscrf result;

            result.fst = ilat::ilat_path_maker()(edges, *f.fst);
            result.topo_order_cache = fst::topo_order(*result.fst);
            result.weight_func = f.weight_func;
            result.feature_func = f.feature_func;
            result.cost_func = f.cost_func;

            return std::make_shared<iscrf>(result);
        }

        composite_feature<ilat::fst, dense_vec> make_feat(
            feat_dim_alloc& alloc,
            std::vector<std::string> features,
            std::vector<std::vector<double>> const& frames,
            std::unordered_map<std::string, std::string> const& args)
        {
            composite_feature<ilat::fst, dense_vec> result;

            using feat_func = segment_feature<ilat::fst, dense_vec, ilat_lexicalizer>;

            for (auto& k: features) {
                if (ebt::startswith(k, "length-indicator")) {
                    std::vector<std::string> parts = ebt::split(k, "@");
                    int order = 0;
                    if (parts.size() > 1) {
                        order = std::stoi(parts[1]);
                    }

                    if (!ebt::in(std::string("max-seg"), args)) {
                        std::cerr << "--max-seg is required" << std::endl;
                        exit(1);
                    }

                    int max_seg = std::stoi(args.at("max-seg"));

                    result.features.push_back(std::make_shared<feat_func>(
                        feat_func(alloc, order,
                        std::make_shared<segfeat::length_indicator>(
                            segfeat::length_indicator { max_seg }),
                        frames)));
                } else if (ebt::startswith(k, "bias")) {
                    std::vector<std::string> parts = ebt::split(k, "@");
                    int order = 0;
                    if (parts.size() > 1) {
                        order = std::stoi(parts[1]);
                    }

                    result.features.push_back(std::make_shared<feat_func>(
                        feat_func(alloc, order,
                        std::make_shared<segfeat::bias>(
                            segfeat::bias {}),
                        frames)));
                } else if (ebt::startswith(k, "frame-avg")
                        || ebt::startswith(k, "frame-samples")
                        || ebt::startswith(k, "left-boundary")
                        || ebt::startswith(k, "right-boundary")) {
                    // leave it for later
                } else {
                    std::cerr << "unknown feature " << k << std::endl;
                    exit(1);
                }
            }

            return result;
        }

        composite_feature_with_frame_grad<ilat::fst, dense_vec> make_feat_with_frame_grad(
            feat_dim_alloc& alloc,
            std::vector<std::string> features,
            std::vector<std::vector<double>> const& frames,
            std::unordered_map<std::string, std::string> const& args)
        {
            composite_feature_with_frame_grad<ilat::fst, dense_vec> result;

            using feat_func = segment_feature_with_frame_grad<ilat::fst, dense_vec, ilat_lexicalizer>;

            for (auto& k: features) {
                if (ebt::startswith(k, "frame-avg")) {
                    std::vector<std::string> parts = ebt::split(k, "@");
                    int order = 0;
                    if (parts.size() > 1) {
                        order = std::stoi(parts[1]);
                    }

                    int start_dim = -1;
                    int end_dim = -1;
                    std::tie(start_dim, end_dim) = get_dim(parts[0]);

                    result.features.push_back(std::make_shared<feat_func>(
                        feat_func(alloc, order,
                        std::make_shared<segfeat::frame_avg>(
                            segfeat::frame_avg { frames, start_dim, end_dim }),
                        frames)));
                } else if (ebt::startswith(k, "frame-samples")) {
                    std::vector<std::string> parts = ebt::split(k, "@");
                    int order = 0;
                    if (parts.size() > 1) {
                        order = std::stoi(parts[1]);
                    }

                    int start_dim = -1;
                    int end_dim = -1;
                    std::tie(start_dim, end_dim) = get_dim(parts[0]);

                    result.features.push_back(std::make_shared<feat_func>(
                        feat_func(alloc, order,
                        std::make_shared<segfeat::frame_samples>(
                            segfeat::frame_samples { 3, start_dim, end_dim }),
                        frames)));
                } else if (ebt::startswith(k, "left-boundary")) {
                    std::vector<std::string> parts = ebt::split(k, "@");
                    int order = 0;
                    if (parts.size() > 1) {
                        order = std::stoi(parts[1]);
                    }

                    int start_dim = -1;
                    int end_dim = -1;
                    std::tie(start_dim, end_dim) = get_dim(parts[0]);

                    result.features.push_back(std::make_shared<feat_func>(
                        feat_func(alloc, order,
                        std::make_shared<segfeat::left_boundary>(
                            segfeat::left_boundary { start_dim, end_dim }),
                        frames)));
                } else if (ebt::startswith(k, "right-boundary")) {
                    std::vector<std::string> parts = ebt::split(k, "@");
                    int order = 0;
                    if (parts.size() > 1) {
                        order = std::stoi(parts[1]);
                    }

                    int start_dim = -1;
                    int end_dim = -1;
                    std::tie(start_dim, end_dim) = get_dim(parts[0]);

                    result.features.push_back(std::make_shared<feat_func>(
                        feat_func(alloc, order,
                        std::make_shared<segfeat::right_boundary>(
                            segfeat::right_boundary { start_dim, end_dim }),
                        frames)));
                }
            }

            return result;
        }

        void parse_inference_args(inference_args& i_args,
            std::unordered_map<std::string, std::string> const& args)
        {
            i_args.args = args;

            i_args.min_seg = 1;
            if (ebt::in(std::string("min-seg"), args)) {
                i_args.min_seg = std::stoi(args.at("min-seg"));
            }

            i_args.max_seg = 20;
            if (ebt::in(std::string("max-seg"), args)) {
                i_args.max_seg = std::stoi(args.at("max-seg"));
            }

            i_args.param = load_dense_vec(args.at("param"));

            std::tie(i_args.nn_param, i_args.pred_param) = load_lstm_param(args.at("nn-param"));

            i_args.features = ebt::split(args.at("features"), ",");

            i_args.label_id = load_label_id(args.at("label"));

            i_args.id_label.resize(i_args.label_id.size());
            for (auto& p: i_args.label_id) {
                i_args.labels.push_back(p.second);
                i_args.id_label[p.second] = p.first;
            }

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

        sample::sample(inference_args const& i_args)
            : graph_alloc(i_args.labels)
        {}

        void make_graph(sample& s, inference_args const& i_args)
        {
            s.graph.fst = ::scrf::make_graph(s.frames.size(),
                i_args.label_id, i_args.id_label, i_args.min_seg, i_args.max_seg);
            s.graph.topo_order_cache = ::fst::topo_order(*s.graph.fst);
        }

        void make_lattice(ilat::fst const& lat, sample& s, inference_args const& i_args)
        {
            s.graph.fst = std::make_shared<ilat::fst>(lat);
            s.graph.topo_order_cache = ::fst::topo_order(lat);
        }

        void parse_learning_args(learning_args& l_args,
            std::unordered_map<std::string, std::string> const& args)
        {
            parse_inference_args(l_args, args);

            l_args.opt_data = load_dense_vec(args.at("opt-data"));

            std::tie(l_args.nn_opt_data, l_args.pred_opt_data)
                = load_lstm_param(args.at("nn-opt-data"));

            l_args.step_size = std::stod(args.at("step-size"));

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

            l_args.cost_scale = 1;
            if (ebt::in(std::string("cost-scale"), args)) {
                l_args.cost_scale = std::stod(args.at("cost-scale"));
                assert(l_args.cost_scale >= 0);
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

        learning_sample::learning_sample(learning_args const& args)
            : sample(args), gold_alloc(args.labels)
        {}

        void make_min_cost_gold(learning_sample& s, learning_args const& l_args)
        {
            auto old_weight_func = s.graph.weight_func;

            s.graph.weight_func = std::make_shared<mul<ilat::fst>>(mul<ilat::fst>(
                std::make_shared<seg_cost<ilat::fst>>(
                    make_overlap_cost<ilat::fst>(s.gold_segs, l_args.sils)),
                -1));

            s.gold = ::fst::shortest_path<iscrf, iscrf_path_maker>(s.graph);

            s.graph.weight_func = old_weight_func;
        }

        void parameterize(iscrf& scrf, feat_dim_alloc& alloc,
            std::vector<std::vector<double>> const& frames,
            inference_args const& i_args)
        {
            using comp_feat = composite_feature<ilat::fst, dense_vec>;
            using comp_feat_with_frame_grad = composite_feature_with_frame_grad<ilat::fst, dense_vec>;

            comp_feat feat_func
                = ::scrf::e2e::make_feat(alloc, i_args.features, frames, i_args.args);
            comp_feat_with_frame_grad feat_func_with_frame_grad
                = make_feat_with_frame_grad(alloc, i_args.features, frames, i_args.args);
            for (auto& f: feat_func_with_frame_grad.features) {
                feat_func.features.push_back(f);
            }

            composite_weight<ilat::fst> weight;
            weight.weights.push_back(std::make_shared<linear_score<ilat::fst, dense_vec>>(
                linear_score<ilat::fst, dense_vec>(i_args.param,
                std::make_shared<comp_feat>(feat_func))));
            scrf.weight_func = std::make_shared<composite_weight<ilat::fst>>(weight);
            scrf.feature_func = std::make_shared<comp_feat>(feat_func);
            scrf.feature_func_with_frame_grad = std::make_shared<comp_feat_with_frame_grad>(
                feat_func_with_frame_grad);
        }

        void parameterize(learning_sample& s, learning_args const& l_args)
        {
            parameterize(s.graph, s.graph_alloc, s.frames, l_args);
            parameterize(*s.gold, s.gold_alloc, s.frames, l_args);
        }

    }

}
