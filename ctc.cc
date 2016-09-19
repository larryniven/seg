#include "seg/ctc.h"
#include "seg/util.h"
#include <fstream>

namespace ctc {

    ilat::fst make_frame_fst(std::vector<std::vector<double>> const& feat,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label)
    {
        assert(feat.front().size() + 1 == id_label.size());

        ilat::fst_data data;
        data.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(label_id);
        data.id_symbol = std::make_shared<std::vector<std::string>>(id_label);

        int u = 0;
        ilat::add_vertex(data, u, ilat::vertex_data { u });

        for (int i = 0; i < feat.size(); ++i) {
            int v = data.vertices.size();
            ilat::add_vertex(data, v, ilat::vertex_data { v });

            for (int d = 0; d < feat[i].size(); ++d) {
                int e = data.edges.size();
                ilat::add_edge(data, e, ilat::edge_data { u, v, feat[i][d], d + 1, d + 1 });
            }

            u = v;
        }

        data.initials.push_back(0);
        data.finals.push_back(data.vertices.size() - 1);

        ilat::fst f;
        f.data = std::make_shared<ilat::fst_data>(data);

        return f;
    }

    ilat::fst make_label_fst(std::vector<std::string> const& label_seq,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label)
    {
        ilat::fst_data data;

        data.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(label_id);
        data.id_symbol = std::make_shared<std::vector<std::string>>(id_label);

        int u = 0;
        ilat::add_vertex(data, u, ilat::vertex_data { u });

        for (int i = 0; i < label_seq.size(); ++i) {
            int v1 = data.vertices.size();
            ilat::add_vertex(data, v1, ilat::vertex_data { v1 });

            int e = data.edges.size();
            ilat::add_edge(data, e, ilat::edge_data { u, v1, 0,
                label_id.at("<blk>"), label_id.at("<blk>") });

            e = data.edges.size();
            ilat::add_edge(data, e, ilat::edge_data { v1, v1, 0,
                label_id.at("<blk>"), label_id.at("<blk>") });

            int v2 = data.vertices.size();
            ilat::add_vertex(data, v2, ilat::vertex_data { v2 });

            e = data.edges.size();
            ilat::add_edge(data, e, ilat::edge_data { u, v2, 0,
                label_id.at(label_seq[i]), label_id.at(label_seq[i]) });

            e = data.edges.size();
            ilat::add_edge(data, e, ilat::edge_data { v1, v2, 0,
                label_id.at(label_seq[i]), label_id.at(label_seq[i]) });

            e = data.edges.size();
            ilat::add_edge(data, e, ilat::edge_data { v2, v2, 0,
                label_id.at(label_seq[i]), label_id.at(label_seq[i]) });

            u = v2;
        }

        int v = 0;
        ilat::add_vertex(data, v, ilat::vertex_data { v });

        int e = data.edges.size();
        ilat::add_edge(data, e, ilat::edge_data { u, v, 0,
            label_id.at("<blk>"), label_id.at("<blk>") });

        e = data.edges.size();
        ilat::add_edge(data, e, ilat::edge_data { v, v, 0,
            label_id.at("<blk>"), label_id.at("<blk>") });

        data.initials.push_back(0);
        data.finals.push_back(data.vertices.size() - 1);

        ilat::fst f;
        f.data = std::make_shared<ilat::fst_data>(data);

        return f;
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

    void parse_inference_args(inference_args& i_args,
        std::unordered_map<std::string, std::string> const& args)
    {
        i_args.args = args;

        std::tie(i_args.layer, i_args.nn_param, i_args.pred_param)
            = load_lstm_param(args.at("param"));

        i_args.dropout = 0;
        if (ebt::in(std::string("dropout"), args)) {
            i_args.dropout = std::stod(args.at("dropout"));
            assert(0 <= i_args.dropout && i_args.dropout <= 1);
        }

        i_args.label_id = util::load_label_id(args.at("label"));

        i_args.id_label.resize(i_args.label_id.size());
        for (auto& p: i_args.label_id) {
            i_args.labels.push_back(p.second);
            i_args.id_label[p.second] = p.first;
        }
    }

    void parse_learning_args(learning_args& l_args,
        std::unordered_map<std::string, std::string> const& args)
    {
        parse_inference_args(l_args, args);

        std::tie(l_args.layer, l_args.nn_opt_data, l_args.pred_opt_data)
            = load_lstm_param(args.at("opt-data"));

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

        l_args.dropout_seed = 0;
        if (ebt::in(std::string("dropout-seed"), args)) {
            l_args.dropout_seed = std::stoi(args.at("dropout-seed"));
        }

        l_args.clip = 0;
        if (ebt::in(std::string("clip"), args)) {
            l_args.clip = std::stod(args.at("clip"));
        }
    }

    std::vector<std::shared_ptr<autodiff::op_t>>
    make_feat(autodiff::computation_graph& comp_graph,
        std::shared_ptr<tensor_tree::vertex> lstm_var_tree,
        std::shared_ptr<tensor_tree::vertex> pred_var_tree,
        lstm::stacked_bi_lstm_nn_t& nn,
        rnn::pred_nn_t& pred_nn,
        std::vector<std::vector<double>> const& frames,
        std::default_random_engine& gen,
        inference_args& nn_args)
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

        pred_nn = rnn::make_pred_nn(pred_var_tree, nn.layer.back().output);

        return pred_nn.logprob;
    }
}
