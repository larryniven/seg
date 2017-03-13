#include "seg/ctc.h"
#include "seg/util.h"
#include "nn/lstm-tensor-tree.h"
#include "fst/ifst.h"
#include "seg/seg-weight.h"
#include <fstream>

namespace ctc {

    ifst::fst make_frame_fst(int nframes,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label)
    {
        ifst::fst_data data;
        data.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(label_id);
        data.id_symbol = std::make_shared<std::vector<std::string>>(id_label);

        int u = 0;
        ifst::add_vertex(data, u, ifst::vertex_data { u });

        for (int i = 0; i < nframes; ++i) {
            int v = data.vertices.size();
            ifst::add_vertex(data, v, ifst::vertex_data { v });

            for (int d = 1; d < id_label.size(); ++d) {
                int e = data.edges.size();
                ifst::add_edge(data, e, ifst::edge_data { u, v, 0, d, d });
            }

            u = v;
        }

        data.initials.push_back(0);
        data.finals.push_back(u);

        ifst::fst f;
        f.data = std::make_shared<ifst::fst_data>(data);

        return f;
    }

    ifst::fst make_label_fst(std::vector<std::string> const& label_seq,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label)
    {
        ifst::fst_data data;
        data.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(label_id);
        data.id_symbol = std::make_shared<std::vector<std::string>>(id_label);

        int u = 0;
        ifst::add_vertex(data, u, ifst::vertex_data { u });

        for (int i = 0; i < label_seq.size(); ++i) {
            int v1 = data.vertices.size();
            ifst::add_vertex(data, v1, ifst::vertex_data { v1 });

            int e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { u, v1, 0,
                label_id.at("<blk>"), label_id.at("<blk>") });

            e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { v1, v1, 0,
                label_id.at("<blk>"), label_id.at("<blk>") });

            int v2 = data.vertices.size();
            ifst::add_vertex(data, v2, ifst::vertex_data { v2 });

            e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { u, v2, 0,
                label_id.at(label_seq[i]), label_id.at(label_seq[i]) });

            e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { v1, v2, 0,
                label_id.at(label_seq[i]), label_id.at(label_seq[i]) });

            e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { v2, v2, 0,
                label_id.at(label_seq[i]), label_id.at(label_seq[i]) });

            u = v2;
        }

        int v = data.vertices.size();
        ifst::add_vertex(data, v, ifst::vertex_data { v });

        int e = data.edges.size();
        ifst::add_edge(data, e, ifst::edge_data { u, v, 0,
            label_id.at("<blk>"), label_id.at("<blk>") });

        e = data.edges.size();
        ifst::add_edge(data, e, ifst::edge_data { v, v, 0,
            label_id.at("<blk>"), label_id.at("<blk>") });

        data.initials.push_back(0);
        data.finals.push_back(u);
        data.finals.push_back(v);

        ifst::fst f;
        f.data = std::make_shared<ifst::fst_data>(data);

        return f;
    }

    label_weight::label_weight(std::vector<std::shared_ptr<autodiff::op_t>> const& label_score)
        : label_score(label_score)
    {}

    double label_weight::operator()(ifst::fst const& f, int e) const
    {
        int label = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        la::tensor_like<double>& prob = autodiff::get_output<la::tensor_like<double>>(
            label_score[tail_time]);
        return prob({label});
    }

    void label_weight::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        int label = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        la::tensor_like<double>& prob = autodiff::get_output<la::tensor_like<double>>(
            label_score[tail_time]);

        if (label_score[tail_time]->grad == nullptr) {
            la::tensor<double> z;
            z.resize(prob.sizes());
            label_score[tail_time]->grad = std::make_shared<la::tensor<double>>(z);
        }

        la::tensor_like<double>& z = autodiff::get_grad<la::tensor_like<double>>(label_score[tail_time]);

        z({label}) += g;
    }

    loss_func::loss_func(seg::iseg_data const& graph_data,
            std::vector<std::string> const& label_seq)
        : graph_data(graph_data)
    {
        label_graph = make_label_fst(label_seq, *graph_data.fst->data->symbol_id, *graph_data.fst->data->id_symbol);

        fst::lazy_pair_mode2_fst<ifst::fst, ifst::fst> pair_fst(label_graph, *graph_data.fst);

        pair_data.fst = std::make_shared<fst::lazy_pair_mode2_fst<ifst::fst, ifst::fst>>(pair_fst);
        pair_data.weight_func = std::make_shared<seg::mode2_weight>(seg::mode2_weight(graph_data.weight_func));

        seg::seg_fst<seg::pair_iseg_data> pair_graph { pair_data };

        auto topo_order = fst::topo_order(pair_graph);

        forward.merge(pair_graph, topo_order);

        auto rev_topo_order = topo_order;
        std::reverse(rev_topo_order.begin(), rev_topo_order.end());

        backward.merge(pair_graph, rev_topo_order);

        double inf = std::numeric_limits<double>::infinity();

        double forward_sum = -inf;

        for (auto& f: pair_fst.finals()) {
            if (!ebt::in(f, forward.extra)) {
                continue;
            }

            forward_sum = ebt::log_add(forward_sum, forward.extra.at(f));
        }

        double backward_sum = -inf;

        for (auto& i: pair_fst.initials()) {
            if (!ebt::in(i, backward.extra)) {
                continue;
            }

            backward_sum = ebt::log_add(backward_sum, backward.extra.at(i));
        }

        std::cout << "forward: " << forward_sum << std::endl;
        std::cout << "backward: " << backward_sum << std::endl;

        logZ = forward_sum;
    }

    double loss_func::loss() const
    {
        return -logZ;
    }

    void loss_func::grad(double scale) const
    {
        seg::seg_fst<seg::pair_iseg_data> pair_graph { pair_data };

        for (auto& e: pair_graph.edges()) {
            if (!ebt::in(pair_graph.tail(e), forward.extra)
                    || !ebt::in(pair_graph.head(e), backward.extra)) {
                continue;
            }

            double g = forward.extra.at(pair_graph.tail(e)) + pair_graph.weight(e)
                + backward.extra.at(pair_graph.head(e)) - logZ;

            pair_data.weight_func->accumulate_grad(-std::exp(g), *pair_data.fst, e);
        }
    }

}
