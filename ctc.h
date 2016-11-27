#ifndef CTC_H
#define CTC_H

#include "seg/ilat.h"
#include "la/la.h"
#include "nn/tensor-tree.h"
#include "nn/lstm.h"
#include "nn/pred.h"

namespace ctc {

    ilat::fst make_frame_fst(std::vector<std::vector<double>> const& feat,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label);

    ilat::fst make_label_fst(std::vector<std::string> const& label_seq,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label);

    struct inference_args {
        std::shared_ptr<tensor_tree::vertex> nn_param;
        std::shared_ptr<tensor_tree::vertex> pred_param;
        int layer;
        double dropout;
        std::unordered_map<std::string, int> label_id;
        std::vector<std::string> id_label;
        std::vector<int> labels;
        std::unordered_map<std::string, std::string> args;
    };

    std::tuple<int, std::shared_ptr<tensor_tree::vertex>, std::shared_ptr<tensor_tree::vertex>>
    load_lstm_param(std::string filename);

    void save_lstm_param(std::shared_ptr<tensor_tree::vertex> nn_param,
        std::shared_ptr<tensor_tree::vertex> pred_param,
        std::string filename);

    void parse_inference_args(inference_args& i_args,
        std::unordered_map<std::string, std::string> const& args);

    struct learning_args
        : public inference_args {

        std::shared_ptr<tensor_tree::vertex> nn_opt_data;
        std::shared_ptr<tensor_tree::vertex> pred_opt_data;
        double clip;
        int dropout_seed;
        double step_size;
        double momentum;
        double decay;
    };

    void parse_learning_args(learning_args& l_args,
        std::unordered_map<std::string, std::string> const& args);

    std::vector<std::shared_ptr<autodiff::op_t>>
    make_feat(autodiff::computation_graph& comp_graph,
        std::shared_ptr<tensor_tree::vertex> lstm_tree_var,
        std::shared_ptr<tensor_tree::vertex> pred_tree_var,
        lstm::stacked_bi_lstm_nn_t& nn,
        rnn::pred_nn_t& pred_nn,
        std::vector<std::vector<double>> const& frames,
        std::default_random_engine& gen,
        inference_args& i_args);

}

#endif
