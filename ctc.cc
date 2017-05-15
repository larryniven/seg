#include "seg/ctc.h"
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

            for (int d = 0; d < id_label.size(); ++d) {
                if (id_label.at(d) == "<eps>") {
                    continue;
                }

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
                label_seq[i], label_seq[i] });

            e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { v1, v2, 0,
                label_seq[i], label_seq[i] });

            e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { v2, v2, 0,
                label_seq[i], label_seq[i] });

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

    ifst::fst make_label_fst_hmm1s(std::vector<int> const& label_seq,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label)
    {
        ifst::fst_data data;
        data.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(label_id);
        data.id_symbol = std::make_shared<std::vector<std::string>>(id_label);

        int u = 0;
        ifst::add_vertex(data, u, ifst::vertex_data { u });

        for (int i = 0; i < label_seq.size(); ++i) {
            int v2 = data.vertices.size();
            ifst::add_vertex(data, v2, ifst::vertex_data { v2 });

            int e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { u, v2, 0,
                label_seq[i], label_seq[i] });

            e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { v2, v2, 0,
                label_seq[i], label_seq[i] });

            u = v2;
        }

        data.initials.push_back(0);
        data.finals.push_back(u);

        ifst::fst f;
        f.data = std::make_shared<ifst::fst_data>(data);

        return f;
    }

    ifst::fst make_label_fst_hmm2s(std::vector<int> const& label_seq,
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
                label_seq[i], label_seq[i] });

            e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { v1, v1, 0,
                label_id.at(id_label.at(label_seq[i]) + "-"), label_id.at(id_label.at(label_seq[i]) + "-") });

            u = v1;
        }

        data.initials.push_back(0);
        data.finals.push_back(u);

        ifst::fst f;
        f.data = std::make_shared<ifst::fst_data>(data);

        return f;
    }

    ifst::fst make_phone_fst(std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label)
    {
        ifst::fst_data data;
        data.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(label_id);
        data.id_symbol = std::make_shared<std::vector<std::string>>(id_label);

        int start = data.vertices.size();
        ifst::add_vertex(data, start, ifst::vertex_data { start });

        int end = data.vertices.size();
        ifst::add_vertex(data, end, ifst::vertex_data { -1 });

        int blk = label_id.at("<blk>");
        int eps = label_id.at("<eps>");

        for (int i = 0; i < id_label.size(); ++i) {
            if (id_label[i] == "<eps>" || id_label[i] == "<blk>") {
                continue;
            }

            int u = data.vertices.size();
            ifst::add_vertex(data, u, ifst::vertex_data { u });

            int e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { start, u, 0, eps, i });

            int v1 = data.vertices.size();
            ifst::add_vertex(data, v1, ifst::vertex_data { v1 });

            e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { u, v1, 0, blk, eps });

            e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { v1, v1, 0, blk, eps });

            int v2 = data.vertices.size();
            ifst::add_vertex(data, v2, ifst::vertex_data { v2 });

            e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { u, v2, 0, i, eps });

            e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { v1, v2, 0, i, eps });

            e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { v2, v2, 0, i, eps });

            int v3 = data.vertices.size();
            ifst::add_vertex(data, v3, ifst::vertex_data { v3 });

            e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { v2, v3, 0, blk, eps });

            e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { v3, v3, 0, blk, eps });

            e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { v2, end, 0, eps, eps });

            e = data.edges.size();
            ifst::add_edge(data, e, ifst::edge_data { v3, end, 0, eps, eps });
        }

        int e = data.edges.size();
        ifst::add_edge(data, e, ifst::edge_data { end, start, 0, eps, eps });

        data.vertices[end].time = data.vertices.size() + 1;

        data.initials.push_back(start);
        data.finals.push_back(end);

        ifst::fst result;
        result.data = std::make_shared<ifst::fst_data>(data);

        return result;
    }

    label_weight::label_weight(std::vector<std::shared_ptr<autodiff::op_t>> const& label_score)
        : label_score(label_score)
    {}

    double label_weight::operator()(ifst::fst const& f, int e) const
    {
        if (f.output(e) == 0) {
            return 0;
        }

        int label = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        la::cpu::tensor_like<double>& prob = autodiff::get_output<la::cpu::tensor_like<double>>(
            label_score[tail_time]);

        return prob({label});
    }

    void label_weight::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        if (f.output(e) == 0) {
            return;
        }

        int label = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        la::cpu::tensor_like<double>& prob = autodiff::get_output<la::cpu::tensor_like<double>>(
            label_score[tail_time]);

        if (label_score[tail_time]->grad == nullptr) {
            la::cpu::tensor<double> z;
            z.resize(prob.sizes());
            label_score[tail_time]->grad = std::make_shared<la::cpu::tensor<double>>(z);
        }

        la::cpu::tensor_like<double>& z = autodiff::get_grad<la::cpu::tensor_like<double>>(
            label_score[tail_time]);

        z({label}) += g;
    }

    loss_func::loss_func(seg::iseg_data const& graph_data,
             ifst::fst const& label_fst)
        : graph_data(graph_data)
    {
        fst::lazy_pair_mode2_fst<ifst::fst, ifst::fst> pair_fst(label_fst, *graph_data.fst);

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
