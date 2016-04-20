#ifndef ISCRF_E2E_H
#define ISCRF_E2E_H

#include "iscrf.h"

namespace scrf {

    namespace e2e {

        struct iscrf
            : public ::scrf::iscrf
            , public with_frame_grad<int, dense_vec> {

            std::shared_ptr<scrf_feature<ilat::fst, dense_vec>> feature_func;
            std::shared_ptr<scrf_feature_with_frame_grad<ilat::fst, dense_vec>> feature_func_with_frame_grad;

            virtual void feature(dense_vec& f, int e) const override;

            virtual void frame_grad(std::vector<std::vector<double>>& f,
                dense_vec const& param, int e) const;
        };

        struct iscrf_path_maker
            : public fst::path_maker<iscrf> {

            virtual std::shared_ptr<iscrf> operator()(std::vector<int> const& edges,
                iscrf const& f) const override;
        };

        composite_feature<ilat::fst, dense_vec> make_feat(
            feat_dim_alloc& alloc,
            std::vector<std::string> features,
            std::vector<std::vector<double>> const& frames,
            std::unordered_map<std::string, std::string> const& args);

        composite_feature_with_frame_grad<ilat::fst, dense_vec> make_feat_with_frame_grad(
            feat_dim_alloc& alloc,
            std::vector<std::string> features,
            std::vector<std::vector<double>> const& frames,
            std::unordered_map<std::string, std::string> const& args);

        struct inference_args
            : public ::scrf::inference_args {

            lstm::dblstm_feat_param_t nn_param;
            rnn::pred_param_t pred_param;
            int subsample_freq;
            int subsample_shift;
            double rnndrop_prob;
        };

        void parse_inference_args(inference_args& i_args,
            std::unordered_map<std::string, std::string> const& args);

        struct sample {
            std::vector<std::vector<double>> frames;

            feat_dim_alloc graph_alloc;

            iscrf graph;
            std::shared_ptr<iscrf> graph_path;

            sample(inference_args const& i_args);
        };

        void make_graph(sample& s, inference_args const& i_args);

        void make_lattice(ilat::fst const& lat, sample& s, inference_args const& i_args);

        struct learning_args
            : public inference_args {

            dense_vec opt_data;
            lstm::dblstm_feat_param_t nn_opt_data;
            rnn::pred_param_t pred_opt_data;
            double step_size;
            double momentum;
            double decay;
            std::vector<int> sils;
            double cost_scale;
            int rnndrop_seed;
        };

        void parse_learning_args(learning_args& l_args,
            std::unordered_map<std::string, std::string> const& args);

        std::tuple<lstm::dblstm_feat_param_t, rnn::pred_param_t>
        load_lstm_param(std::string filename);

        void save_lstm_param(lstm::dblstm_feat_param_t const& nn_param,
            rnn::pred_param_t const& pred_param,
            std::string filename);

        struct learning_sample
            : public sample {

            std::vector<segcost::segment<int>> gold_segs;

            feat_dim_alloc gold_alloc;

            std::shared_ptr<iscrf> gold;

            std::shared_ptr<scrf_weight<ilat::fst>> cost;

            learning_sample(learning_args const& args);
        };

        void make_min_cost_gold(learning_sample& s, learning_args const& l_args);

        void parameterize(iscrf& scrf, feat_dim_alloc& alloc,
            std::vector<std::vector<double>> const& frames,
            inference_args const& i_args);

        void parameterize(learning_sample& s, learning_args const& l_args);
    }

}

#endif
