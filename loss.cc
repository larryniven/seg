#include "seg/loss.h"
#include "seg/seg-util.h"
#include "ebt/ebt.h"
#include "seg/seg-weight.h"

namespace seg {

    loss_func::~loss_func()
    {}

    marginal_log_loss::marginal_log_loss(iseg_data& graph_data,
        std::vector<int> const& label_seq)
        : graph_data(graph_data)
    {
        seg_fst<iseg_data> graph { graph_data };

        forward_graph.merge(graph, *graph_data.topo_order);

        auto rev_topo_order = *graph_data.topo_order;
        std::reverse(rev_topo_order.begin(), rev_topo_order.end());

        backward_graph.merge(graph, rev_topo_order);

        for (auto& f: graph.finals()) {
            std::cout << "forward: " << forward_graph.extra[f] << std::endl;
        }

        for (auto& i: graph.initials()) {
            std::cout << "backward: " << backward_graph.extra[i] << std::endl;
        }

        ifst::fst& graph_fst = *graph_data.fst;
        auto& label_id = *graph_fst.data->symbol_id;
        auto& id_label = *graph_fst.data->id_symbol;

        ifst::fst label_fst = make_label_fst(label_seq, label_id, id_label);

        fst::lazy_pair_mode1_fst<ifst::fst, ifst::fst> composed_fst { label_fst, graph_fst };

        pair_data.fst = std::make_shared<fst::lazy_pair_mode1_fst<ifst::fst, ifst::fst>>(composed_fst);
        pair_data.weight_func = std::make_shared<mode2_weight>(
            mode2_weight { graph_data.weight_func });
        pair_data.topo_order = std::make_shared<std::vector<std::tuple<int, int>>>(
            fst::topo_order(composed_fst));

        seg_fst<pair_iseg_data> pair { pair_data };

        forward_label.merge(pair, *pair_data.topo_order);

        std::vector<std::tuple<int, int>> rev_pair_topo_order = *pair_data.topo_order;
        std::reverse(rev_pair_topo_order.begin(), rev_pair_topo_order.end());
        backward_label.merge(pair, rev_pair_topo_order);

        for (auto& f: pair.finals()) {
            std::cout << "forward: " << forward_label.extra[f] << std::endl;
        }

        for (auto& i: pair.initials()) {
            std::cout << "backward: " << backward_label.extra[i] << std::endl;
        }

    }

    double marginal_log_loss::loss() const
    {
        double result = 0;

        seg_fst<pair_iseg_data> pair { pair_data };

        result -= forward_label.extra.at(pair.finals().front());

        seg_fst<iseg_data> graph { graph_data };

        result += forward_graph.extra.at(graph.finals().front());

        return result;
    }

    void marginal_log_loss::grad() const
    {
        seg_fst<pair_iseg_data> pair { pair_data };

        double logZ1 = forward_label.extra.at(pair.finals().front());

        for (auto& e: pair.edges()) {
            if (!ebt::in(pair.tail(e), forward_label.extra) ||
                    !ebt::in(pair.head(e), backward_label.extra)) {
                continue;
            }

            pair_data.weight_func->accumulate_grad(
                -std::exp(forward_label.extra.at(pair.tail(e)) + pair.weight(e)
                    + backward_label.extra.at(pair.head(e)) - logZ1), *pair_data.fst, e);
        }

        seg_fst<iseg_data> graph { graph_data };

        double logZ2 = forward_graph.extra.at(graph.finals().front());

        for (auto& e: graph.edges()) {
            graph_data.weight_func->accumulate_grad(
                std::exp(forward_graph.extra.at(graph.tail(e)) + graph.weight(e)
                    + backward_graph.extra.at(graph.head(e)) - logZ2), *graph_data.fst, e);
        }
    }

    entropy_loss::entropy_loss(iseg_data& graph_data)
        : graph_data(graph_data)
    {
        seg_fst<iseg_data> graph { graph_data };

        forward_graph.merge(graph, *graph_data.topo_order);

        std::vector<int> rev_topo_order = *graph_data.topo_order;
        std::reverse(rev_topo_order.begin(), rev_topo_order.end());
        backward_graph.merge(graph, rev_topo_order);

        double inf = std::numeric_limits<double>::infinity();

        double forward_logZ = -inf;
        for (auto& f: graph.finals()) {
            forward_logZ = ebt::log_add(forward_logZ, forward_graph.extra.at(f));
        }

        double backward_logZ = -inf;
        for (auto& i: graph.initials()) {
            backward_logZ = ebt::log_add(backward_logZ, backward_graph.extra.at(i));
        }

        std::cout << "forward: " << forward_logZ << " backward: " << backward_logZ << std::endl;

        logZ = forward_logZ;

        forward_exp.merge(graph, *graph_data.topo_order, logZ, forward_graph.extra);
        backward_exp.merge(graph, rev_topo_order, logZ, backward_graph.extra);

        double forward_exp_score = 0;
        for (auto& f: graph.finals()) {
            forward_exp_score += forward_exp.extra.at(f);
        }

        double backward_exp_score = 0;
        for (auto& i: graph.initials()) {
            backward_exp_score += backward_exp.extra.at(i);
        }

        std::cout << "forward: " << forward_exp_score << " backward: " << backward_exp_score << std::endl;

        exp_score = forward_exp_score;
    }

    double entropy_loss::loss() const
    {
        return -logZ + exp_score;
    }

    void entropy_loss::grad() const
    {
        seg_fst<iseg_data> graph { graph_data };

        for (auto& e: graph.edges()) {
            int tail = graph.tail(e);
            int head = graph.head(e);
            double weight = graph.weight(e);

            double e_marginal = std::exp(forward_graph.extra.at(tail)
                + weight + backward_graph.extra.at(head) - logZ);

            double e_exp = std::exp(weight - logZ) * (
                forward_exp.extra.at(tail) * std::exp(backward_graph.extra.at(head))
                + std::exp(forward_graph.extra.at(tail) + weight + backward_graph.extra.at(head))
                + std::exp(forward_graph.extra.at(tail)) * backward_exp.extra.at(head));

            graph_data.weight_func->accumulate_grad(-exp_score * e_marginal + e_exp, *graph_data.fst, e);
        }
    }

}
