#ifndef SEG_UTIL_H
#define SEG_UTIL_H

#include "fst/ifst.h"
#include "autodiff/autodiff.h"
#include "nn/tensor-tree.h"
#include "seg/seg-weight.h"
#include "nn/lstm.h"
#include <vector>
#include <unordered_map>
#include <random>
#include "seg/cost.h"

namespace seg {

    ifst::fst make_label_fst(std::vector<int> const& label_seq,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label);

    ifst::fst make_label_fst_1b(std::vector<int> const& label_seq,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label);

    ifst::fst make_label_fst(std::vector<int> const& label_seq,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label,
        std::vector<std::string> const& long_labels);

    ifst::fst make_forward_label_fst(std::vector<int> const& label_seq,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label);

    std::shared_ptr<ifst::fst> make_graph(int frames,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label,
        int min_seg_len, int max_seg_len, int stride);

    std::shared_ptr<ifst::fst> make_forward_graph(int frames,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label,
        int min_seg_len, int max_seg_len, int stride);

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(
        std::vector<std::string> const& features);

    std::shared_ptr<seg_weight<ifst::fst>> make_weights(
        std::vector<std::string> const& features,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> frame_mat,
        double dropout = 0.0,
        std::default_random_engine *gen = nullptr);

    struct inference_args {
        int min_seg;
        int max_seg;
        int stride;
        std::shared_ptr<tensor_tree::vertex> param;
        std::unordered_map<std::string, int> label_id;
        std::vector<std::string> id_label;
        std::vector<int> labels;
        std::vector<std::string> features;
        std::unordered_map<std::string, std::string> args;

        std::default_random_engine gen;
    };

    std::shared_ptr<tensor_tree::vertex> make_lstm_tensor_tree(
        int outer_layer, int inner_layer);

    std::tuple<int, int, std::shared_ptr<tensor_tree::vertex>>
    load_lstm_param(std::string filename);

    std::shared_ptr<lstm::transcriber>
    make_transcriber(
        int outer_layer,
        int inner_layer,
        std::unordered_map<std::string, std::string> const& args,
        std::default_random_engine *gen);

    void save_lstm_param(
        std::shared_ptr<tensor_tree::vertex> nn_param,
        std::string filename);

    void save_lstm_param(
        int outer_layer, int inner_layer,
        std::shared_ptr<tensor_tree::vertex> nn_param,
        std::string filename);

    void parse_inference_args(inference_args& l_args,
        std::unordered_map<std::string, std::string> const& args);

    struct sample {
        std::vector<std::vector<double>> frames;
        iseg_data graph_data;

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

        std::vector<cost::segment<int>> gt_segs;
        iseg_data gold_data;

        learning_sample(learning_args const& args);
    };

}

#endif
