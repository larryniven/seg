#include "scrf/iscrf_segnn.h"
#include <fstream>

namespace iscrf {

    namespace segnn {

        void parse_nn_inference_args(nn_inference_args& i_args,
            std::unordered_map<std::string, std::string> const& args)
        {
            std::ifstream ifs { args.at("nn-param") };
            i_args.nn_param = ::segnn::load_nn_param(ifs);
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

            std::ifstream ifs { args.at("nn-opt-data") };
            l_args.nn_opt_data = ::segnn::load_nn_param(ifs);
        }

        ::segnn::segnn_feat make_segnn_feat(
            std::vector<std::string> features,
            std::vector<std::vector<double>> const& frames,
            nn_inference_args const& i_args,
            std::unordered_map<std::string, std::string> const& args)
        {
            segfeat::composite_feat segfeats;
            int dim = frames.front().size();

            for (auto& k: features) {
                if (ebt::startswith(k, "segnn-frame-avg")) {
                    segfeats.features.push_back(std::make_shared<segfeat::frame_avg>(
                        segfeat::frame_avg { frames, 0, dim }));
                } else if (ebt::startswith(k, "segnn-frame-samples")) {
                    segfeats.features.push_back(std::make_shared<segfeat::frame_samples>(
                        segfeat::frame_samples { 3, 0, dim }));
                } else if (ebt::startswith(k, "segnn-left-boundary")) {
                    segfeats.features.push_back(std::make_shared<segfeat::left_boundary>(
                        segfeat::left_boundary { 0, dim }));
                } else if (ebt::startswith(k, "segnn-right-boundary")) {
                    segfeats.features.push_back(std::make_shared<segfeat::right_boundary>(
                        segfeat::right_boundary { 0, dim }));
                } else if (ebt::startswith(k, "segnn-length-indicator")) {
                    int max_seg = std::stoi(args.at("max-seg"));

                    segfeats.features.push_back(std::make_shared<segfeat::length_indicator>(
                        segfeat::length_indicator { max_seg }));
                }
            }

            return ::segnn::segnn_feat(frames, std::make_shared<segfeat::composite_feat>(segfeats),
                i_args.nn_param, i_args.nn);
        }

        void parameterize(iscrf_data& data,
            ::segnn::segnn_feat const& segnn_feat,
            ::iscrf::inference_args const& i_args)
        {
            using comp_feat = scrf::composite_feature<ilat::fst, scrf::dense_vec>;

            comp_feat feat_func;
            feat_func.features.push_back(std::make_shared<::segnn::segnn_feat>(segnn_feat));

            scrf::composite_weight<ilat::fst> weight;
            weight.weights.push_back(std::make_shared<scrf::linear_score<ilat::fst, scrf::dense_vec>>(
                scrf::linear_score<ilat::fst, scrf::dense_vec>(i_args.param,
                std::make_shared<comp_feat>(feat_func))));

            data.weight_func = std::make_shared<scrf::composite_weight<ilat::fst>>(weight);
            data.feature_func = std::make_shared<comp_feat>(feat_func);
        }

        void parameterize(learning_sample& s, learning_args const& l_args)
        {
            ::segnn::segnn_feat gold_segnn_feat = make_segnn_feat(
                l_args.features, s.frames, l_args, l_args.args);
            ::segnn::segnn_feat graph_segnn_feat = make_segnn_feat(
                l_args.features, s.frames, l_args, l_args.args);

            ::iscrf::segnn::parameterize(s.graph_data, graph_segnn_feat, l_args);

            s.graph_data.cost_func = std::make_shared<scrf::mul<ilat::fst>>(scrf::mul<ilat::fst>(
                std::make_shared<scrf::seg_cost<ilat::fst>>(
                    scrf::make_overlap_cost<ilat::fst>(s.gold_segs, l_args.sils)),
                l_args.cost_scale));

            ::iscrf::segnn::parameterize(s.gold_data, gold_segnn_feat, l_args);

            s.gold_data.cost_func = std::make_shared<scrf::mul<ilat::fst>>(scrf::mul<ilat::fst>(
                std::make_shared<scrf::seg_cost<ilat::fst>>(
                    scrf::make_overlap_cost<ilat::fst>(s.gold_segs, l_args.sils)),
                l_args.cost_scale));
        }

    }

}
