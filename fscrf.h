#ifndef FSCRF_H
#define FSCRF_H

#include "seg/scrf.h"
#include "seg/scrf_weight.h"
#include "seg/segcost.h"
#include "seg/scrf_cost.h"
#include "autodiff/autodiff.h"
#include "nn/tensor-tree.h"
#include "nn/lstm.h"
#include "nn/pred.h"
#include <random>

namespace fscrf {

    struct fscrf_data {
        std::shared_ptr<ilat::fst> fst;
        std::shared_ptr<std::vector<int>> topo_order;
        std::shared_ptr<scrf::scrf_weight<ilat::fst>> weight_func;
        std::shared_ptr<scrf::scrf_weight<ilat::fst>> cost_func;
        std::shared_ptr<tensor_tree::vertex> param;
    };

    using fscrf_fst = scrf::scrf_fst<fscrf_data>;

    struct fscrf_pair_data {
        std::shared_ptr<ilat::pair_fst> fst;
        std::shared_ptr<std::vector<std::tuple<int, int>>> topo_order;
        std::shared_ptr<scrf::scrf_weight<ilat::pair_fst>> weight_func;
        std::shared_ptr<scrf::scrf_weight<ilat::pair_fst>> cost_func;
        std::shared_ptr<tensor_tree::vertex> param;
    };

    using fscrf_pair_fst = scrf::scrf_fst<fscrf_pair_data>;

}

namespace scrf {

    template <>
    struct scrf_data_trait<fscrf::fscrf_data> {
        using base_fst = ilat::fst;
        using path_maker = ilat::ilat_path_maker;
        using edge = int;
        using vertex = int;
        using symbol = int;
        using fst = scrf_fst<fscrf::fscrf_data>;
    };

    template <>
    struct scrf_data_trait<fscrf::fscrf_pair_data> {
        using base_fst = ilat::pair_fst;
        using path_maker = ilat::pair_fst_path_maker;
        using edge = std::tuple<int, int>;
        using vertex = std::tuple<int, int>;
        using symbol = int;
        using fst = scrf_fst<fscrf::fscrf_pair_data>;
    };

}

namespace fscrf {

    std::shared_ptr<ilat::fst> make_graph(int frames,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label,
        int min_seg_len, int max_seg_len, int stride);

    std::shared_ptr<ilat::fst> make_random_graph(int frames,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label,
        int min_seg_len, int max_seg_len, int stride);

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(
        std::vector<std::string> const& features);

    std::shared_ptr<scrf::composite_weight<ilat::fst>> make_weights(
        std::vector<std::string> const& features,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> frame_mat,
        double dropout = 0.0,
        std::default_random_engine *gen = nullptr);

    struct frame_avg_score
        : public scrf::scrf_weight<ilat::fst> {

        std::shared_ptr<autodiff::op_t> param;
        std::shared_ptr<autodiff::op_t> frames;
        std::shared_ptr<autodiff::op_t> score;

        frame_avg_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames);

        virtual double operator()(ilat::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ilat::fst const& f,
            int e) const override;

        virtual void grad() const override;

    };

    struct frame_weighted_avg_score
        : public scrf::scrf_weight<ilat::fst> {

        std::shared_ptr<autodiff::op_t> param;
        std::shared_ptr<autodiff::op_t> att_param;
        std::shared_ptr<autodiff::op_t> frames;
        std::shared_ptr<autodiff::op_t> att;
        std::shared_ptr<autodiff::op_t> att_exp;
        std::shared_ptr<autodiff::op_t> score;

        frame_weighted_avg_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> att_param,
            std::shared_ptr<autodiff::op_t> frame);

        virtual double operator()(ilat::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ilat::fst const& f,
            int e) const override;

        virtual void grad() const override;

    };

    struct frame_samples_score
        : public scrf::scrf_weight<ilat::fst> {

        std::shared_ptr<autodiff::op_t> param;
        std::shared_ptr<autodiff::op_t> frames;
        std::shared_ptr<autodiff::op_t> score;
        double scale;

        frame_samples_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames, double scale);

        virtual double operator()(ilat::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ilat::fst const& f,
            int e) const override;

        virtual void grad() const override;

    };

    struct left_boundary_score
        : public scrf::scrf_weight<ilat::fst> {

        std::shared_ptr<autodiff::op_t> param;
        std::shared_ptr<autodiff::op_t> frames;
        std::shared_ptr<autodiff::op_t> score;
        int shift;

        left_boundary_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames, int shift);

        virtual double operator()(ilat::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ilat::fst const& f,
            int e) const override;

        virtual void grad() const override;

    };

    struct right_boundary_score
        : public scrf::scrf_weight<ilat::fst> {

        std::shared_ptr<autodiff::op_t> param;
        std::shared_ptr<autodiff::op_t> frames;
        std::shared_ptr<autodiff::op_t> score;
        int shift;

        right_boundary_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames, int shift);

        virtual double operator()(ilat::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ilat::fst const& f,
            int e) const override;

        virtual void grad() const override;

    };

    struct segrnn_mod_score
        : public scrf::scrf_weight<ilat::fst> {

        std::shared_ptr<tensor_tree::vertex> param;
        std::shared_ptr<autodiff::op_t> frames;
        std::shared_ptr<autodiff::op_t> pre_left;
        std::shared_ptr<autodiff::op_t> pre_right;
        std::shared_ptr<autodiff::op_t> pre_label;
        std::shared_ptr<autodiff::op_t> pre_length;

        std::shared_ptr<autodiff::op_t> score;

        std::vector<int> topo_order_shift;

        mutable std::default_random_engine *gen;
        double dropout;

        mutable std::vector<std::shared_ptr<autodiff::op_t>> edge_scores;

        segrnn_mod_score(std::shared_ptr<tensor_tree::vertex> param,
            std::shared_ptr<autodiff::op_t> frames);

        segrnn_mod_score(std::shared_ptr<tensor_tree::vertex> param,
            std::shared_ptr<autodiff::op_t> frames,
            double dropout,
            std::default_random_engine *gen);

        virtual double operator()(ilat::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ilat::fst const& f,
            int e) const override;

        virtual void grad() const override;

    };

    struct segrnn_score
        : public scrf::scrf_weight<ilat::fst> {

        std::shared_ptr<tensor_tree::vertex> param;
        std::shared_ptr<autodiff::op_t> frames;
        std::shared_ptr<autodiff::op_t> pre_left;
        std::shared_ptr<autodiff::op_t> pre_right;
        std::shared_ptr<autodiff::op_t> left_end;
        std::shared_ptr<autodiff::op_t> right_end;
        std::shared_ptr<autodiff::op_t> pre_label;
        std::shared_ptr<autodiff::op_t> pre_length;

        std::shared_ptr<autodiff::op_t> score;

        std::vector<int> topo_order_shift;

        mutable std::default_random_engine *gen;
        double dropout;

        mutable std::vector<std::shared_ptr<autodiff::op_t>> edge_scores;

        segrnn_score(std::shared_ptr<tensor_tree::vertex> param,
            std::shared_ptr<autodiff::op_t> frames);

        segrnn_score(std::shared_ptr<tensor_tree::vertex> param,
            std::shared_ptr<autodiff::op_t> frames,
            double dropout,
            std::default_random_engine *gen);

        virtual double operator()(ilat::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ilat::fst const& f,
            int e) const override;

        virtual void grad() const override;

    };

    struct left_boundary_order2_score
        : public scrf::scrf_weight<ilat::pair_fst> {

        std::shared_ptr<tensor_tree::vertex> param;
        std::vector<std::vector<double>> const& frames;
        std::vector<std::shared_ptr<autodiff::op_t>> frames_cat;
        int context;

        std::shared_ptr<autodiff::op_t> score;
        std::shared_ptr<autodiff::op_t> input;
        std::shared_ptr<autodiff::op_t> label_embedding1;
        std::shared_ptr<autodiff::op_t> label_embedding2;
        std::vector<std::shared_ptr<autodiff::op_t>> topo_order;

        left_boundary_order2_score(std::shared_ptr<tensor_tree::vertex> param,
            std::vector<std::vector<double>> const& frames, int context);

        virtual double operator()(ilat::pair_fst const& f,
            std::tuple<int, int> e) const;

        virtual void accumulate_grad(double g, ilat::pair_fst const& f,
            std::tuple<int, int> e) const override;

    };

    struct log_length_score
        : public scrf::scrf_weight<ilat::fst> {

        std::shared_ptr<autodiff::op_t> param;

        log_length_score(std::shared_ptr<autodiff::op_t> param);

        virtual double operator()(ilat::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ilat::fst const& f,
            int e) const override;

    };

    struct length_score
        : public scrf::scrf_weight<ilat::fst> {

        std::shared_ptr<autodiff::op_t> param;

        length_score(std::shared_ptr<autodiff::op_t> param);

        virtual double operator()(ilat::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ilat::fst const& f,
            int e) const override;

    };

    struct bias0_score
        : public scrf::scrf_weight<ilat::fst> {

        std::shared_ptr<autodiff::op_t> param;

        bias0_score(std::shared_ptr<autodiff::op_t> param);

        virtual double operator()(ilat::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ilat::fst const& f,
            int e) const override;

    };

    struct bias1_score
        : public scrf::scrf_weight<ilat::fst> {

        std::shared_ptr<autodiff::op_t> param;

        bias1_score(std::shared_ptr<autodiff::op_t> param);

        virtual double operator()(ilat::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ilat::fst const& f,
            int e) const override;

    };

    struct external_score_order0
        : public scrf::scrf_weight<ilat::fst> {

        std::vector<int> indices;
        std::shared_ptr<autodiff::op_t> param;

        external_score_order0(std::shared_ptr<autodiff::op_t> param,
            std::vector<int> indices);

        virtual double operator()(ilat::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ilat::fst const& f,
            int e) const override;

    };

    struct external_score_order1
        : public scrf::scrf_weight<ilat::fst> {

        std::vector<int> indices;
        std::shared_ptr<autodiff::op_t> param;

        external_score_order1(std::shared_ptr<autodiff::op_t> param,
            std::vector<int> indices);

        virtual double operator()(ilat::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ilat::fst const& f,
            int e) const override;

    };

    struct edge_weight
        : public scrf::scrf_weight<ilat::fst> {

        std::shared_ptr<autodiff::op_t> param;

        edge_weight(std::shared_ptr<autodiff::op_t> param);

        virtual double operator()(ilat::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ilat::fst const& f,
            int e) const override;

    };

    struct inference_args {
        int min_seg;
        int max_seg;
        int stride;
        std::shared_ptr<tensor_tree::vertex> param;
        std::shared_ptr<tensor_tree::vertex> nn_param;
        std::shared_ptr<tensor_tree::vertex> pred_param;
        int inner_layer;
        int outer_layer;
        std::unordered_map<std::string, int> label_id;
        std::vector<std::string> id_label;
        std::vector<int> labels;
        std::vector<std::string> features;
        std::unordered_map<std::string, std::string> args;

        std::default_random_engine gen;
    };

    std::shared_ptr<tensor_tree::vertex> make_lstm_tensor_tree(
        int outer_layer, int inner_layer);

    std::tuple<int, int, std::shared_ptr<tensor_tree::vertex>, std::shared_ptr<tensor_tree::vertex>>
    load_lstm_param(std::string filename);

    std::shared_ptr<lstm::transcriber>
    make_transcriber(inference_args& i_args);

    void save_lstm_param(
        int outer_layer, int inner_layer,
        std::shared_ptr<tensor_tree::vertex> nn_param,
        std::shared_ptr<tensor_tree::vertex> pred_param,
        std::string filename);

    void parse_inference_args(inference_args& l_args,
        std::unordered_map<std::string, std::string> const& args);

    struct sample {
        std::vector<std::vector<double>> frames;
        fscrf_data graph_data;

        sample(inference_args const& i_args);
    };

    void make_graph(sample& s, inference_args& i_args);
    void make_graph(sample& s, inference_args& i_args, int frames);

    struct learning_args
        : public inference_args {

        double cost_scale;
        std::shared_ptr<tensor_tree::vertex> opt_data;
        std::shared_ptr<tensor_tree::vertex> first_moment;
        std::shared_ptr<tensor_tree::vertex> second_moment;
        std::shared_ptr<tensor_tree::vertex> nn_opt_data;
        std::shared_ptr<tensor_tree::vertex> nn_first_moment;
        std::shared_ptr<tensor_tree::vertex> nn_second_moment;
        std::shared_ptr<tensor_tree::vertex> pred_opt_data;
        std::shared_ptr<tensor_tree::vertex> pred_first_moment;
        std::shared_ptr<tensor_tree::vertex> pred_second_moment;
        double l2;
        double step_size;
        double momentum;
        double decay;
        std::vector<int> sils;
        double adam_beta1;
        double adam_beta2;
        int time;
    };

    void parse_learning_args(learning_args& l_args,
        std::unordered_map<std::string, std::string> const& args);

    struct learning_sample
        : public sample {

        std::vector<segcost::segment<int>> gt_segs;
        fscrf_data gold_data;

        learning_sample(learning_args const& args);
    };

    struct loss_func {
        virtual ~loss_func();

        virtual double loss() const = 0;
        virtual void grad() const = 0;
    };

    struct hinge_loss
        : public loss_func {

        fscrf_data& graph_data;

        fscrf_data graph_path_data;
        fscrf_data gold_path_data;

        std::vector<int> const& sils;
        std::vector<segcost::segment<int>> gold_segs;
        double cost_scale;

        hinge_loss(fscrf_data& graph_data,
            std::vector<segcost::segment<int>> const& gt_segs,
            std::vector<int> const& sils,
            double cost_scale);

        virtual double loss() const override;

        virtual void grad() const override;

    };

    struct hinge_loss_gt
        : public loss_func {

        fscrf_data& graph_data;

        fscrf_data graph_path_data;
        fscrf_data gold_path_data;

        std::vector<int> const& sils;
        std::vector<segcost::segment<int>> gold_segs;
        double cost_scale;

        hinge_loss_gt(fscrf_data& graph_data,
            std::vector<segcost::segment<int>> const& gt_segs,
            std::vector<int> const& sils,
            double cost_scale);

        virtual double loss() const override;

        virtual void grad() const override;

    };

    struct log_loss
        : public loss_func {

        fscrf_data& graph_data;
        fscrf_data gold_path_data;
        std::vector<segcost::segment<int>> gold_segs;

        fst::forward_log_sum<fscrf_fst> forward;
        fst::backward_log_sum<fscrf_fst> backward;

        log_loss(fscrf_data& graph_data,
            std::vector<segcost::segment<int>> const& gt_segs,
            std::vector<int> const& sils);

        virtual double loss() const override;

        virtual void grad() const override;

    };

    ilat::fst make_label_fst(std::vector<int> const& label_seq,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label);

    struct marginal_log_loss
        : public loss_func {

        fscrf_data& graph_data;

        fst::forward_log_sum<fscrf_fst> forward_graph;
        fst::backward_log_sum<fscrf_fst> backward_graph;

        fscrf_pair_data pair_data;

        fst::forward_log_sum<fscrf_pair_fst> forward_label;
        fst::backward_log_sum<fscrf_pair_fst> backward_label;

        marginal_log_loss(fscrf_data& graph_data,
            std::vector<int> const& label_seq);

        virtual double loss() const override;

        virtual void grad() const override;

    };

    struct latent_hinge_loss
        : public loss_func {

        fscrf_data& graph_data;
        fscrf_data graph_path_data;

        fscrf_pair_data pair_data;
        fscrf_pair_data gold_path_data;

        std::vector<segcost::segment<int>> gold_segs;
        std::vector<int> const& sils;
        double cost_scale;

        latent_hinge_loss(fscrf_data& graph_data,
            std::vector<int> const& label_seq,
            std::vector<int> const& sils,
            double cost_scale);

        virtual double loss() const override;

        virtual void grad() const override;

    };

    struct mode1_weight
        : public scrf::scrf_weight<ilat::pair_fst> {

        std::shared_ptr<scrf::scrf_weight<ilat::fst>> weight;

        mode1_weight(std::shared_ptr<scrf::scrf_weight<ilat::fst>> weight);

        virtual double operator()(ilat::pair_fst const& fst,
            std::tuple<int, int> e) const override;

        virtual void accumulate_grad(double g, ilat::pair_fst const& fst,
            std::tuple<int, int> e) const override;

        virtual void grad() const override;
    };

    struct mode2_weight
        : public scrf::scrf_weight<ilat::pair_fst> {

        std::shared_ptr<scrf::scrf_weight<ilat::fst>> weight;

        mode2_weight(std::shared_ptr<scrf::scrf_weight<ilat::fst>> weight);

        virtual double operator()(ilat::pair_fst const& fst,
            std::tuple<int, int> e) const override;

        virtual void accumulate_grad(double g, ilat::pair_fst const& fst,
            std::tuple<int, int> e) const override;

        virtual void grad() const override;
    };

    std::shared_ptr<scrf::scrf_weight<ilat::fst>> make_lat_weights(
        std::vector<std::string> const& feature,
        std::shared_ptr<tensor_tree::vertex> var_tree);

    std::shared_ptr<scrf::scrf_weight<ilat::pair_fst>> make_pair_weights(
        std::vector<std::string> const& features,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::vector<double>> const& frames);

    struct hinge_loss_pair
        : public loss_func {

        fscrf_pair_data& graph_data;

        fscrf_pair_data graph_path_data;
        fscrf_pair_data gold_path_data;

        std::vector<int> const& sils;
        std::vector<segcost::segment<int>> gold_segs;
        double cost_scale;

        hinge_loss_pair(fscrf_pair_data& graph_data,
            std::vector<segcost::segment<int>> const& gt_segs,
            std::vector<int> const& sils,
            double cost_scale);

        virtual double loss() const override;

        virtual void grad() const override;

    };

}

#endif
