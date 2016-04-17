#ifndef ISCRF_H
#define ISCRF_H

#include "scrf/experimental/scrf.h"
#include "scrf/experimental/scrf_feat.h"
#include "scrf/experimental/scrf_cost.h"
#include "scrf/experimental/scrf_weight.h"
#include "nn/lstm.h"
#include "nn/pred.h"

namespace scrf {

    struct iscrf
        : public scrf<int, int, int, dense_vec> {

        using base_fst = ilat::fst;
        using vector = dense_vec;
        using vertex = int;
        using edge = int;
        using symbol = int;

        std::shared_ptr<ilat::fst> fst;
        std::vector<vertex> topo_order_cache;

        std::shared_ptr<scrf_weight<ilat::fst>> weight_func;
        std::shared_ptr<scrf_feature<ilat::fst, dense_vec>> feature_func;
        std::shared_ptr<scrf_weight<ilat::fst>> cost_func;

        virtual std::vector<int> const& vertices() const override;
        virtual std::vector<int> const& edges() const override;
        virtual int head(int e) const override;
        virtual int tail(int e) const override;
        virtual std::vector<int> const& in_edges(int v) const override;
        virtual std::vector<int> const& out_edges(int v) const override;
        virtual double weight(int e) const override;
        virtual int const& input(int e) const override;
        virtual int const& output(int e) const override;
        virtual std::vector<int> const& initials() const override;
        virtual std::vector<int> const& finals() const override;

        virtual long time(int v) const override;

        virtual void feature(dense_vec& f, int e) const override;
        virtual double cost(int e) const override;

        virtual std::vector<int> const& topo_order() const override;

    };

    struct iscrf_path_maker
        : public fst::path_maker<iscrf> {

        virtual std::shared_ptr<iscrf> operator()(std::vector<int> const& edges,
            iscrf const& f) const override;
    };

    std::shared_ptr<ilat::fst> make_graph(int frames,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label,
        int min_seg_len, int max_seg_len);

    struct ilat_lexicalizer
        : public lexicalizer<ilat::fst, dense_vec> {

        virtual double* lex(feat_dim_alloc const& alloc,
            int order, dense_vec& f, ilat::fst const& a, int e) const override;
    };

    std::pair<int, int> get_dim(std::string feat);

    composite_feature<ilat::fst, dense_vec> make_feat(
        feat_dim_alloc& alloc,
        std::vector<std::string> features,
        std::vector<std::vector<double>> const& frames,
        std::unordered_map<std::string, std::string> const& args);

    struct inference_args {
        int min_seg;
        int max_seg;
        dense_vec param;
        std::unordered_map<std::string, int> label_id;
        std::vector<std::string> id_label;
        std::vector<int> labels;
        std::vector<std::string> features;
        std::unordered_map<std::string, std::string> args;
    };

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
        double step_size;
        double momentum;
        double decay;
        std::vector<int> sils;
    };

    struct learning_sample
        : public sample {

        ilat::fst ground_truth_fst;

        feat_dim_alloc gold_alloc;

        std::shared_ptr<iscrf> ground_truth;
        std::shared_ptr<iscrf> gold;

        std::shared_ptr<seg_cost<ilat::fst>> cost;

        learning_sample(learning_args const& args);
    };

    learning_args parse_learning_args(
        std::unordered_map<std::string, std::string> const& args);

    void make_gold(learning_sample& s, learning_args const& l_args);

    void make_min_cost_gold(learning_sample& s, learning_args const& l_args);

    void parameterize(iscrf& scrf, feat_dim_alloc& alloc,
        std::vector<std::vector<double>> const& frames,
        inference_args const& i_args);

    void parameterize(learning_sample& s, learning_args const& l_args);

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
            : ::scrf::inference_args {

            lstm::dblstm_feat_param_t nn_param;
            rnn::pred_param_t pred_param;
            int subsample_freq;
            int subsample_shift;
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

            ilat::fst ground_truth_fst;

            feat_dim_alloc gold_alloc;

            std::shared_ptr<iscrf> ground_truth;
            std::shared_ptr<iscrf> gold;

            std::shared_ptr<seg_cost<ilat::fst>> cost;

            learning_sample(learning_args const& args);
        };

        void make_gold(learning_sample& s, learning_args const& l_args);

        void make_min_cost_gold(learning_sample& s, learning_args const& l_args);

        void parameterize(iscrf& scrf, feat_dim_alloc& alloc,
            std::vector<std::vector<double>> const& frames,
            inference_args const& i_args);

        void parameterize(learning_sample& s, learning_args const& l_args);
    }

}


#endif
