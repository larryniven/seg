#include "seg/seg-util.h"
#include <cassert>
#include "nn/lstm-tensor-tree.h"
#include <fstream>
#include "fst/fst-algo.h"
#include "speech/speech.h"
#include "ebt/ebt.h"

using namespace std::string_literals;

namespace seg {

    ifst::fst make_label_fst(std::vector<int> const& label_seq,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label)
    {
        ifst::fst_data data;

        data.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(label_id);
        data.id_symbol = std::make_shared<std::vector<std::string>>(id_label);

        int u = 0;
        ifst::add_vertex(data, u, ifst::vertex_data { u });

        for (int i = 0; i < label_seq.size(); ++i) {
            int v = data.vertices.size();
            ifst::add_vertex(data, v, ifst::vertex_data { v });

            int e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { u, v, 0,
                label_seq[i], label_seq[i] });

            u = v;
        }

        data.initials.push_back(0);
        data.finals.push_back(data.vertices.size() - 1);

        ifst::fst f;
        f.data = std::make_shared<ifst::fst_data>(data);

        return f;
    }

    ifst::fst make_label_fst_1b(std::vector<int> const& label_seq,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label)
    {
        ifst::fst_data data;

        data.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(label_id);
        data.id_symbol = std::make_shared<std::vector<std::string>>(id_label);

        int u = 0;
        ifst::add_vertex(data, u, ifst::vertex_data { u });

        for (int i = 0; i < label_seq.size(); ++i) {
            int v = data.vertices.size();
            ifst::add_vertex(data, v, ifst::vertex_data { v });

            int e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { u, v, 0,
                label_seq[i], label_seq[i] });

            e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { v, v, 0,
                label_id.at("<blk>"), label_id.at("<blk>") });

            u = v;
        }

        data.initials.push_back(0);
        data.finals.push_back(data.vertices.size() - 1);

        ifst::fst f;
        f.data = std::make_shared<ifst::fst_data>(data);

        return f;
    }

    ifst::fst make_label_fst(std::vector<int> const& label_seq,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label,
        std::vector<std::string> const& long_labels)
    {
        ifst::fst_data data;

        data.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(label_id);
        data.id_symbol = std::make_shared<std::vector<std::string>>(id_label);

        std::unordered_set<int> long_label_set;
        for (auto& ell: long_labels) {
            long_label_set.insert(label_id.at(ell));
        }

        int u = 0;
        ifst::add_vertex(data, u, ifst::vertex_data { u });

        for (int i = 0; i < label_seq.size(); ++i) {
            int v = data.vertices.size();
            ifst::add_vertex(data, v, ifst::vertex_data { v });

            int e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { u, v, 0,
                label_seq[i], label_seq[i] });

            if (ebt::in(label_seq[i], long_label_set)) {
                int e = data.edges.size();
                ifst::add_edge(data, e, ifst::edge_data { v, v, 0,
                    label_seq[i], label_seq[i] });
            }

            u = v;
        }

        data.initials.push_back(0);
        data.finals.push_back(data.vertices.size() - 1);

        ifst::fst f;
        f.data = std::make_shared<ifst::fst_data>(data);

        return f;
    }

    std::shared_ptr<ifst::fst> make_graph(int frames,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label,
        int min_seg_len, int max_seg_len, int stride)
    {
        assert(stride >= 1);
        assert(min_seg_len >= 1);
        assert(max_seg_len >= min_seg_len);

        ifst::fst_data data;

        data.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(label_id);
        data.id_symbol = std::make_shared<std::vector<std::string>>(id_label);

        int i = 0;
        int v = -1;
        for (i = 0; i < frames + 1; i += stride) {
            ++v;
            ifst::add_vertex(data, v, ifst::vertex_data { i });
        }

        data.initials.push_back(0);
        data.finals.push_back(v);

        for (int u = 0; u < data.vertices.size(); ++u) {
            for (int v = u + 1; v < data.vertices.size(); ++v) {
                int duration = data.vertices[v].time - data.vertices[u].time;

                if (duration < min_seg_len) {
                    continue;
                }

                if (duration > max_seg_len) {
                    break;
                }

                for (auto& p: label_id) {
                    if (p.first == "<eps>") {
                        continue;
                    }

                    ifst::add_edge(data, data.edges.size(),
                        ifst::edge_data { u, v, 0, p.second, p.second });
                }
            }
        }

        ifst::fst result;
        result.data = std::make_shared<ifst::fst_data>(std::move(data));

        return std::make_shared<ifst::fst>(result);
    }

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(
        std::vector<std::string> const& features)
    {
        tensor_tree::vertex root { "nil"s };

        for (auto& k: features) {
            if (ebt::startswith(k, "ext0")) {
                root.children.push_back(tensor_tree::make_tensor("ext0"));
            } else if (ebt::startswith(k, "ext1")) {
                root.children.push_back(tensor_tree::make_tensor("ext1"));
            } else if (ebt::startswith(k, "frame-avg")) {
                root.children.push_back(tensor_tree::make_tensor("frame avg"));
            } else if (ebt::startswith(k, "frame-att")) {
                root.children.push_back(tensor_tree::make_tensor("frame att"));
                root.children.push_back(tensor_tree::make_tensor("frame att"));
            } else if (ebt::startswith(k, "frame-samples")) {
                root.children.push_back(tensor_tree::make_tensor("frame samples"));
                root.children.push_back(tensor_tree::make_tensor("frame samples"));
                root.children.push_back(tensor_tree::make_tensor("frame samples"));
            } else if (ebt::startswith(k, "left-boundary")) {
                root.children.push_back(tensor_tree::make_tensor("left boundary"));
                root.children.push_back(tensor_tree::make_tensor("left boundary"));
                root.children.push_back(tensor_tree::make_tensor("left boundary"));
            } else if (ebt::startswith(k, "right-boundary")) {
                root.children.push_back(tensor_tree::make_tensor("right boundary"));
                root.children.push_back(tensor_tree::make_tensor("right boundary"));
                root.children.push_back(tensor_tree::make_tensor("right boundary"));
            } else if (ebt::startswith(k, "length-indicator")) {
                root.children.push_back(tensor_tree::make_tensor("length"));
            } else if (k == "lm") {
                root.children.push_back(tensor_tree::make_tensor("lm"));
            } else if (k == "lat-weight") {
                root.children.push_back(tensor_tree::make_tensor("lat-weight"));
            } else if (k == "bias0") {
                root.children.push_back(tensor_tree::make_tensor("bias0"));
            } else if (k == "bias1") {
                root.children.push_back(tensor_tree::make_tensor("bias1"));
            } else if (k == "boundary2") {
                tensor_tree::vertex v { "nil"s };
                v.children.push_back(tensor_tree::make_tensor("left boundary order2 acoustic embedding"));
                v.children.push_back(tensor_tree::make_tensor("left boundary order2 label1 embedding"));
                v.children.push_back(tensor_tree::make_tensor("left boundary order2 label2 embedding"));
                v.children.push_back(tensor_tree::make_tensor("left boundary order2 weight"));
                v.children.push_back(tensor_tree::make_tensor("left boundary order2 weight"));
                root.children.push_back(std::make_shared<tensor_tree::vertex>(v));
            } else if (k == "segrnn") {
                tensor_tree::vertex v { "nil"s };
                v.children.push_back(tensor_tree::make_tensor("segrnn left embedding"));
                v.children.push_back(tensor_tree::make_tensor("segrnn left end"));
                v.children.push_back(tensor_tree::make_tensor("segrnn right embedding"));
                v.children.push_back(tensor_tree::make_tensor("segrnn right end"));
                v.children.push_back(tensor_tree::make_tensor("segrnn label embedding"));
                v.children.push_back(tensor_tree::make_tensor("segrnn label embedding"));
                v.children.push_back(tensor_tree::make_tensor("segrnn length embedding"));
                v.children.push_back(tensor_tree::make_tensor("segrnn length embedding"));
                v.children.push_back(tensor_tree::make_tensor("segrnn bias1"));
                v.children.push_back(tensor_tree::make_tensor("segrnn weight1"));
                v.children.push_back(tensor_tree::make_tensor("segrnn bias2"));
                v.children.push_back(tensor_tree::make_tensor("segrnn weight2"));
                root.children.push_back(std::make_shared<tensor_tree::vertex>(v));
            } else if (k == "logsoftmax") {
                root.children.push_back(tensor_tree::make_tensor("softmax"));
            } else {
                std::cout << "unknown feature " << k << std::endl;
                exit(1);
            }
        }

        return std::make_shared<tensor_tree::vertex>(root);
    }

    std::shared_ptr<seg_weight<ifst::fst>> make_weights(
        std::vector<std::string> const& features,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> frame_mat,
        double dropout,
        std::default_random_engine *gen)
    {
        composite_weight<ifst::fst> weight_func;

        int feat_idx = 0;

        for (auto& k: features) {
            if (ebt::startswith(k, "frame-avg")) {
                weight_func.weights.push_back(std::make_shared<frame_avg_score>(
                    frame_avg_score(tensor_tree::get_var(var_tree->children[feat_idx]), frame_mat)));

                ++feat_idx;
            } else if (ebt::startswith(k, "frame-samples")) {
                weight_func.weights.push_back(std::make_shared<frame_samples_score>(
                    frame_samples_score(tensor_tree::get_var(var_tree->children[feat_idx]), frame_mat, 1.0 / 6)));
                weight_func.weights.push_back(std::make_shared<frame_samples_score>(
                    frame_samples_score(tensor_tree::get_var(var_tree->children[feat_idx + 1]), frame_mat, 1.0 / 2)));
                weight_func.weights.push_back(std::make_shared<frame_samples_score>(
                    frame_samples_score(tensor_tree::get_var(var_tree->children[feat_idx + 2]), frame_mat, 5.0 / 6)));

                feat_idx += 3;
            } else if (ebt::startswith(k, "left-boundary")) {
                weight_func.weights.push_back(std::make_shared<left_boundary_score>(
                    left_boundary_score(tensor_tree::get_var(var_tree->children[feat_idx]), frame_mat, -1)));
                weight_func.weights.push_back(std::make_shared<left_boundary_score>(
                    left_boundary_score(tensor_tree::get_var(var_tree->children[feat_idx + 1]), frame_mat, -2)));
                weight_func.weights.push_back(std::make_shared<left_boundary_score>(
                    left_boundary_score(tensor_tree::get_var(var_tree->children[feat_idx + 2]), frame_mat, -3)));

                feat_idx += 3;
            } else if (ebt::startswith(k, "right-boundary")) {
                weight_func.weights.push_back(std::make_shared<right_boundary_score>(
                    right_boundary_score(tensor_tree::get_var(var_tree->children[feat_idx]), frame_mat, 1)));
                weight_func.weights.push_back(std::make_shared<right_boundary_score>(
                    right_boundary_score(tensor_tree::get_var(var_tree->children[feat_idx + 1]), frame_mat, 2)));
                weight_func.weights.push_back(std::make_shared<right_boundary_score>(
                    right_boundary_score(tensor_tree::get_var(var_tree->children[feat_idx + 2]), frame_mat, 3)));

                feat_idx += 3;
            } else if (ebt::startswith(k, "length-indicator")) {
                weight_func.weights.push_back(std::make_shared<length_score>(
                    length_score { tensor_tree::get_var(var_tree->children[feat_idx]) }));

                ++feat_idx;
            } else if (k == "bias0") {
                weight_func.weights.push_back(std::make_shared<bias0_score>(
                    bias0_score { tensor_tree::get_var(var_tree->children[feat_idx]) }));

                ++feat_idx;
            } else if (k == "bias1") {
                weight_func.weights.push_back(std::make_shared<bias1_score>(
                    bias1_score { tensor_tree::get_var(var_tree->children[feat_idx]) }));

                ++feat_idx;
            } else if (ebt::startswith(k, "segrnn")) {
                weight_func.weights.push_back(std::make_shared<segrnn_score>(
                    segrnn_score(var_tree->children[feat_idx], frame_mat, dropout, gen)));

                ++feat_idx;
            } else if (ebt::startswith(k, "logsoftmax")) {
                weight_func.weights.push_back(std::make_shared<logsoftmax_score>(
                    logsoftmax_score(tensor_tree::get_var(var_tree->children[feat_idx]), frame_mat)));

                ++feat_idx;
            } else {
                std::cout << "unknown feature: " << k << std::endl;
                exit(1);
            }
        }

        return std::make_shared<cached_weight<ifst::fst>>(cached_weight<ifst::fst>(
            std::make_shared<composite_weight<ifst::fst>>(weight_func)));
    }

    std::shared_ptr<tensor_tree::vertex> make_lstm_tensor_tree(
        int outer_layer, int inner_layer)
    {
        tensor_tree::vertex root { "nil"s };

        if (inner_layer == -1) {
            lstm::multilayer_lstm_tensor_tree_factory fac {
                std::make_shared<lstm::bi_lstm_tensor_tree_factory>(
                      lstm::bi_lstm_tensor_tree_factory {
                          std::make_shared<lstm::dyer_lstm_tensor_tree_factory>(
                              lstm::dyer_lstm_tensor_tree_factory{})
                          // std::make_shared<lstm::lstm_tensor_tree_factory>(
                          //     lstm::lstm_tensor_tree_factory{})
                      }
                ), outer_layer};

            root.children.push_back(fac());
        } else {
            lstm::multilayer_lstm_tensor_tree_factory fac {
                std::make_shared<lstm::bi_lstm_tensor_tree_factory>(
                lstm::bi_lstm_tensor_tree_factory {
                    std::make_shared<lstm::multilayer_lstm_tensor_tree_factory>(
                    lstm::multilayer_lstm_tensor_tree_factory {
                       std::make_shared<lstm::dyer_lstm_tensor_tree_factory>(
                       lstm::dyer_lstm_tensor_tree_factory{}),
                       inner_layer
                    })
                }), outer_layer};

            root.children.push_back(fac());
        }

        root.children.push_back(tensor_tree::make_tensor("softmax weight"));
        root.children.push_back(tensor_tree::make_tensor("softmax bias"));

        return std::make_shared<tensor_tree::vertex>(root);
    }

    std::tuple<int, int, std::shared_ptr<tensor_tree::vertex>>
    load_lstm_param(std::string filename)
    {
        std::ifstream ifs { filename };
        std::string line;

        std::getline(ifs, line);
        auto parts = ebt::split(line);

        if (parts.size() == 1) {
            int layer = std::stoi(line);
            std::shared_ptr<tensor_tree::vertex> nn_param = make_lstm_tensor_tree(layer, -1);
            tensor_tree::load_tensor(nn_param, ifs);

            return std::make_tuple(layer, -1, nn_param);
        } else if (parts.size() == 2) {
            int outer_layer = std::stoi(parts[0]);
            int inner_layer = std::stoi(parts[1]);
            std::shared_ptr<tensor_tree::vertex> nn_param = make_lstm_tensor_tree(outer_layer, inner_layer);
            tensor_tree::load_tensor(nn_param, ifs);

            return std::make_tuple(outer_layer, inner_layer, nn_param);
        }
    }

    void save_lstm_param(
        std::shared_ptr<tensor_tree::vertex> nn_param,
        std::string filename)
    {
        std::ofstream ofs { filename };

        ofs << nn_param->children.size() << std::endl;
        tensor_tree::save_tensor(nn_param, ofs);
    }

    void save_lstm_param(int outer_layer, int inner_layer,
        std::shared_ptr<tensor_tree::vertex> nn_param,
        std::string filename)
    {
        std::ofstream ofs { filename };

        if (inner_layer == -1) {
            ofs << nn_param->children.size() << std::endl;
        } else {
            ofs << outer_layer << " " << inner_layer << std::endl;
        }
        tensor_tree::save_tensor(nn_param, ofs);
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

        i_args.stride = 1;
        if (ebt::in(std::string("stride"), args)) {
            i_args.stride = std::stoi(args.at("stride"));
        }

        if (ebt::in(std::string("features"), args)) {
            i_args.features = ebt::split(args.at("features"), ",");
        }

        if (ebt::in(std::string("param"), args)) {
            i_args.param = make_tensor_tree(i_args.features);

            tensor_tree::load_tensor(i_args.param, args.at("param"));
        }

        i_args.label_id = speech::load_label_id(args.at("label"));

        i_args.id_label.resize(i_args.label_id.size());
        for (auto& p: i_args.label_id) {
            i_args.labels.push_back(p.second);
            i_args.id_label[p.second] = p.first;
        }

        if (ebt::in(std::string("seed"), args)) {
           i_args.gen = std::default_random_engine { std::stoul(args.at("seed")) };
        }
    }

    sample::sample(inference_args const& i_args)
    {
        graph_data.param = i_args.param;
    }

    void make_graph(sample& s, inference_args& i_args, int frames)
    {
        s.graph_data.fst = make_graph(frames,
            i_args.label_id, i_args.id_label, i_args.min_seg, i_args.max_seg, i_args.stride);
        s.graph_data.topo_order = std::make_shared<std::vector<int>>(
            fst::topo_order(*s.graph_data.fst));
    }

    void make_graph(sample& s, inference_args& i_args)
    {
        make_graph(s, i_args, s.frames.size());
    }

    void parse_learning_args(learning_args& l_args,
        std::unordered_map<std::string, std::string> const& args)
    {
        parse_inference_args(l_args, args);

        if (ebt::in(std::string("opt-data"), args)) {
            l_args.opt_data = make_tensor_tree(l_args.features);
            tensor_tree::load_tensor(l_args.opt_data, args.at("opt-data"));
        }

        l_args.l2 = 0;
        if (ebt::in(std::string("l2"), args)) {
            l_args.l2 = std::stod(args.at("l2"));
        }

        l_args.step_size = 0;
        if (ebt::in(std::string("step-size"), args)) {
            l_args.step_size = std::stod(args.at("step-size"));
        }

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

        if (ebt::in(std::string("sil"), l_args.label_id)) {
            l_args.sils.push_back(l_args.label_id.at("sil"));
        }

        if (ebt::in(std::string("adam-beta1"), args)) {
            l_args.adam_beta1 = std::stod(args.at("adam-beta1"));
        }

        if (ebt::in(std::string("adam-beta2"), args)) {
            l_args.adam_beta2 = std::stod(args.at("adam-beta2"));
        }
    }

    learning_sample::learning_sample(learning_args const& l_args)
        : sample(l_args)
    {
        gold_data.param = l_args.param;
    }

}
