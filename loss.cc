#include "seg/loss.h"
#include "seg/seg-util.h"
#include "ebt/ebt.h"
#include "seg/seg-weight.h"

namespace seg {

    loss_func::~loss_func()
    {}

    hinge_loss::hinge_loss(iseg_data& graph_data,
        std::vector<cost::segment<int>> const& gt_segs,
        std::vector<int> const& sils, double cost_scale)
        : graph_data(graph_data)
        , cost_scale(cost_scale)
    {
        seg_fst<iseg_data> graph { graph_data };

        cost::overlap_cost<int> cost_func { sils };

        auto old_weight_func = graph_data.weight_func;

        graph_data.weight_func = make_weight<ifst::fst>([&](ifst::fst const& f, int e) {
            int tail_time = graph.time(graph.tail(e));
            int head_time = graph.time(graph.head(e));
            cost::segment<int> s { tail_time, head_time, graph.output(e) };
            return -cost_func(gt_segs, s);
        });

        fst::forward_one_best<seg_fst<iseg_data>> min_cost_one_best;
        for (auto& i: graph.initials()) {
            min_cost_one_best.extra[i] = {-1, 0};
        }
        min_cost_one_best.merge(graph, *graph_data.topo_order);
        min_cost_path = min_cost_one_best.best_path(graph);

        for (auto& e: min_cost_path) {
            int tail_time = graph.time(graph.tail(e));
            int head_time = graph.time(graph.head(e));

            min_cost_segs.push_back(cost::segment<int> { tail_time, head_time, graph.output(e) });
        }

        auto& id_symbol = *graph_data.fst->data->id_symbol;

        double gold_cost = 0;
        gold_weight = 0;
        std::cout << "gold:";
        for (auto& e: min_cost_path) {
            int tail_time = graph.time(graph.tail(e));
            int head_time = graph.time(graph.head(e));
            cost::segment<int> s { tail_time, head_time, graph.output(e) };
            double c = cost_func(gt_segs, s);
            gold_cost += c;
            gold_weight += (*old_weight_func)(*graph_data.fst, e);

            std::cout << " " << id_symbol[graph.output(e)] << " (" << c << ")";
        }
        std::cout << std::endl;
        std::cout << "gold cost: " << gold_cost << std::endl;
        std::cout << "gold weight: " << gold_weight << std::endl;

        graph_data.weight_func = make_weight<ifst::fst>([&](ifst::fst const& f, int e) {
            int tail_time = graph.time(graph.tail(e));
            int head_time = graph.time(graph.head(e));
            cost::segment<int> s { tail_time, head_time, graph.output(e) };
            return cost_func(min_cost_segs, s) * cost_scale + (*old_weight_func)(f, e);
        });

        fst::forward_one_best<seg_fst<iseg_data>> one_best;
        for (auto& i: graph.initials()) {
            one_best.extra[i] = {-1, 0};
        }
        one_best.merge(graph, *graph_data.topo_order);
        cost_aug_path = one_best.best_path(graph);

        cost_aug_cost = 0;
        cost_aug_weight = 0;
        std::cout << "cost aug inf:";
        for (auto& e: cost_aug_path) {
            int tail_time = graph.time(graph.tail(e));
            int head_time = graph.time(graph.head(e));
            cost::segment<int> s { tail_time, head_time, graph.output(e) };
            double c = cost_func(min_cost_segs, s);
            cost_aug_cost += c;
            cost_aug_weight += (*old_weight_func)(*graph_data.fst, e);

            std::cout << " " << id_symbol[graph.output(e)] << " (" << c << ")";
        }
        std::cout << std::endl;
        std::cout << "cost aug path cost: " << cost_aug_cost * cost_scale << std::endl;
        std::cout << "cost aug path weight: " << cost_aug_weight << std::endl;

        graph_data.weight_func = old_weight_func;
    }

    double hinge_loss::loss() const
    {
        return cost_aug_cost * cost_scale - gold_weight + cost_aug_weight;
    }

    void hinge_loss::grad(double scale) const
    {
        seg_fst<iseg_data> graph { graph_data };

        for (auto& e: min_cost_path) {
            graph_data.weight_func->accumulate_grad(-scale, *graph_data.fst, e);
        }

        for (auto& e: cost_aug_path) {
            graph_data.weight_func->accumulate_grad(scale, *graph_data.fst, e);
        }
    }

    log_loss::log_loss(iseg_data& graph_data,
        std::vector<cost::segment<int>> const& gt_segs,
        std::vector<int> const& sils)
        : graph_data(graph_data)
    {
        seg_fst<iseg_data> graph { graph_data };

        cost::overlap_cost<int> cost_func { sils };

        auto old_weight_func = graph_data.weight_func;

        graph_data.weight_func = make_weight<ifst::fst>([&](ifst::fst const& f, int e) {
            int tail_time = graph.time(graph.tail(e));
            int head_time = graph.time(graph.head(e));
            cost::segment<int> s { tail_time, head_time, graph.output(e) };
            return -cost_func(gt_segs, s);
        });

        fst::forward_one_best<seg_fst<iseg_data>> one_best;
        for (auto& i: graph.initials()) {
            one_best.extra[i] = {-1, 0};
        }
        one_best.merge(graph, *graph_data.topo_order);
        min_cost_path = one_best.best_path(graph);

        graph_data.weight_func = old_weight_func;

        for (auto& e: min_cost_path) {
            int tail_time = graph.time(graph.tail(e));
            int head_time = graph.time(graph.head(e));

            min_cost_segs.push_back(cost::segment<int> { tail_time, head_time, graph.output(e) });
        }

        auto& id_symbol = *graph_data.fst->data->id_symbol;

        double gold_cost = 0;
        double gold_score = 0;
        std::cout << "gold:";
        for (auto& e: min_cost_path) {
            int tail_time = graph.time(graph.tail(e));
            int head_time = graph.time(graph.head(e));
            cost::segment<int> s { tail_time, head_time, graph.output(e) };
            double c = cost_func(gt_segs, s);
            gold_cost += c;
            gold_score += graph.weight(e);

            std::cout << " " << id_symbol[graph.output(e)] << " (" << c << ")";
        }
        std::cout << std::endl;
        std::cout << "gold cost: " << gold_cost << std::endl;
        std::cout << "gold score: " << gold_score << std::endl;

        forward.merge(graph, *graph_data.topo_order);

        auto rev_topo_order = *graph_data.topo_order;
        std::reverse(rev_topo_order.begin(), rev_topo_order.end());

        backward.merge(graph, rev_topo_order);

        double inf = std::numeric_limits<double>::infinity();

        double forward_logZ = -inf;

        for (auto& f: graph.finals()) {
            forward_logZ = ebt::log_add(forward_logZ, forward.extra.at(f));
        }

        double backward_logZ = -inf;

        for (auto& i: graph.initials()) {
            backward_logZ = ebt::log_add(backward_logZ, backward.extra.at(i));
        }

        logZ = forward_logZ;

        std::cout << "forward: " << forward_logZ << std::endl;
        std::cout << "backward: " << backward_logZ << std::endl;
    }

    double log_loss::loss() const
    {
        double result = 0;

        seg_fst<iseg_data> graph { graph_data };

        for (auto& e: min_cost_path) {
            result -= graph.weight(e);
        }

        result += logZ;

        return result;
    }

    void log_loss::grad(double scale) const
    {
        seg_fst<iseg_data> graph { graph_data };

        for (auto& e: min_cost_path) {
            graph_data.weight_func->accumulate_grad(-scale, *graph_data.fst, e);
        }

        for (auto& e: graph.edges()) {
            int tail = graph.tail(e);
            int head = graph.head(e);

            if (!ebt::in(tail, forward.extra) || !ebt::in(head, backward.extra)) {
                continue;
            }

            graph_data.weight_func->accumulate_grad(
                scale * std::exp(forward.extra.at(tail) + graph.weight(e)
                    + backward.extra.at(head) - logZ), *graph_data.fst, e);
        }
    }

    marginal_log_loss::marginal_log_loss(iseg_data& graph_data,
        ifst::fst& label_fst)
        : graph_data(graph_data)
    {
        seg_fst<iseg_data> graph { graph_data };

        forward_graph.merge(graph, *graph_data.topo_order);

        auto rev_topo_order = *graph_data.topo_order;
        std::reverse(rev_topo_order.begin(), rev_topo_order.end());

        backward_graph.merge(graph, rev_topo_order);

        double inf = std::numeric_limits<double>::infinity();

        double f_graph_logZ = -inf;

        for (auto& f: graph.finals()) {
            f_graph_logZ = ebt::log_add(f_graph_logZ, forward_graph.extra[f]);
        }

        std::cout << "forward: " << f_graph_logZ << std::endl;

        double b_graph_logZ = -inf;

        for (auto& i: graph.initials()) {
            b_graph_logZ = ebt::log_add(b_graph_logZ, backward_graph.extra[i]);
        }

        std::cout << "backward: " << b_graph_logZ << std::endl;

        graph_logZ = f_graph_logZ;

        ifst::fst& graph_fst = *graph_data.fst;

        fst::lazy_pair_mode2_fst<ifst::fst, ifst::fst> composed_fst { label_fst, graph_fst };

        pair_data.fst = std::make_shared<fst::lazy_pair_mode2_fst<ifst::fst, ifst::fst>>(composed_fst);
        pair_data.weight_func = std::make_shared<mode2_weight>(
            mode2_weight { graph_data.weight_func });
        pair_data.topo_order = std::make_shared<std::vector<std::tuple<int, int>>>(
            fst::topo_order(composed_fst));

        seg_fst<pair_iseg_data> pair { pair_data };

        forward_label.merge(pair, *pair_data.topo_order);

        std::vector<std::tuple<int, int>> rev_pair_topo_order = *pair_data.topo_order;
        std::reverse(rev_pair_topo_order.begin(), rev_pair_topo_order.end());
        backward_label.merge(pair, rev_pair_topo_order);

        double f_label_logZ = -inf;

        for (auto& f: pair.finals()) {
            f_label_logZ = ebt::log_add(f_label_logZ, forward_label.extra[f]);
        }

        std::cout << "forward: " << f_label_logZ << std::endl;

        double b_label_logZ = -inf;

        for (auto& i: pair.initials()) {
            b_label_logZ = ebt::log_add(b_label_logZ, backward_label.extra[i]);
        }

        std::cout << "backward: " << b_label_logZ << std::endl;

        label_logZ = f_label_logZ;
    }

    double marginal_log_loss::loss() const
    {
        return -label_logZ + graph_logZ;
    }

    void marginal_log_loss::grad(double scale) const
    {
        seg_fst<pair_iseg_data> pair { pair_data };

        for (auto& e: pair.edges()) {
            if (!ebt::in(pair.tail(e), forward_label.extra) ||
                    !ebt::in(pair.head(e), backward_label.extra)) {
                continue;
            }

            pair_data.weight_func->accumulate_grad(
                scale * (-std::exp(forward_label.extra.at(pair.tail(e)) + pair.weight(e)
                    + backward_label.extra.at(pair.head(e)) - label_logZ)), *pair_data.fst, e);
        }

        seg_fst<iseg_data> graph { graph_data };

        for (auto& e: graph.edges()) {
            graph_data.weight_func->accumulate_grad(
                scale * (std::exp(forward_graph.extra.at(graph.tail(e)) + graph.weight(e)
                    + backward_graph.extra.at(graph.head(e)) - graph_logZ)), *graph_data.fst, e);
        }
    }

    double weight_risk::operator()(seg_fst<iseg_data> const& f, int e) const
    {
        return f.weight(e);
    }

    entropy_loss::entropy_loss(iseg_data& graph_data)
        : graph_data(graph_data)
        , forward_exp(std::make_shared<weight_risk>(weight_risk{}))
        , backward_exp(std::make_shared<weight_risk>(weight_risk{}))
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

        forward_exp.merge(graph, *graph_data.topo_order, forward_graph.extra);
        backward_exp.merge(graph, rev_topo_order, backward_graph.extra);

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
        return logZ - exp_score;
    }

    void entropy_loss::grad(double scale) const
    {
        seg_fst<iseg_data> graph { graph_data };

        for (auto& e: graph.edges()) {
            int tail = graph.tail(e);
            int head = graph.head(e);
            double weight = graph.weight(e);

            if (!ebt::in(tail, forward_graph.extra) || !ebt::in(head, backward_graph.extra)) {
                continue;
            }

            double e_marginal = std::exp(forward_graph.extra.at(tail)
                + backward_graph.extra.at(head) - logZ + weight);

            double e_exp = forward_exp.extra.at(tail) + weight + backward_exp.extra.at(head);

            if (std::isinf(e_marginal) || std::isnan(e_marginal)) {
                std::cout << forward_graph.extra.at(tail)
                    << " " << backward_graph.extra.at(head)
                    << " " << logZ
                    << " " << weight << std::endl;
                exit(1);
            }

            if (std::isinf(e_exp) || std::isnan(e_exp)) {
                std::cout << " " << forward_exp.extra.at(tail)
                    << " " << weight
                    << " " << backward_exp.extra.at(head) << std::endl;
                exit(1);
            }

            graph_data.weight_func->accumulate_grad(scale * ((exp_score - e_exp) * e_marginal), *graph_data.fst, e);
        }
    }

    empirical_bayes_risk::empirical_bayes_risk(iseg_data& graph_data,
        std::shared_ptr<risk_func<seg_fst<iseg_data>>> risk)
        : graph_data(graph_data) , risk(risk) , f_risk(risk) , b_risk(risk)
    {
        seg_fst<iseg_data> graph { graph_data };

        f_log_sum.merge(graph, *graph_data.topo_order);

        std::vector<int> rev_topo_order = *graph_data.topo_order;
        std::reverse(rev_topo_order.begin(), rev_topo_order.end());

        b_log_sum.merge(graph, rev_topo_order);

        double inf = std::numeric_limits<double>::infinity();

        double forward_logZ = -inf;
        for (auto& f: graph.finals()) {
            forward_logZ = ebt::log_add(forward_logZ, f_log_sum.extra.at(f));
        }

        double backward_logZ = -inf;
        for (auto& i: graph.initials()) {
            backward_logZ = ebt::log_add(backward_logZ, b_log_sum.extra.at(i));
        }

        std::cout << "forward: " << forward_logZ << " backward: " << backward_logZ << std::endl;

        logZ = forward_logZ;

        f_risk.merge(graph, *graph_data.topo_order, f_log_sum.extra);
        b_risk.merge(graph, rev_topo_order, b_log_sum.extra);

        double forward_exp_risk = 0;
        for (auto& f: graph.finals()) {
            forward_exp_risk += f_risk.extra.at(f);
        }

        double backward_exp_risk = 0;
        for (auto& i: graph.initials()) {
            backward_exp_risk += b_risk.extra.at(i);
        }

        std::cout << "forward: " << forward_exp_risk << " backward: " << backward_exp_risk << std::endl;

        exp_risk = forward_exp_risk;
    }

    double empirical_bayes_risk::loss() const
    {
        return exp_risk;
    }

    void empirical_bayes_risk::grad(double scale) const
    {
        seg_fst<iseg_data> graph { graph_data };

        for (auto& e: graph.edges()) {
            int tail = graph.tail(e);
            int head = graph.head(e);
            double weight = graph.weight(e);
            double r_e = (*risk)(graph, e);

            if (!ebt::in(tail, f_log_sum.extra) || !ebt::in(head, b_log_sum.extra)) {
                continue;
            }

            double e_marginal = std::exp(f_log_sum.extra.at(tail)
                + b_log_sum.extra.at(head) - logZ + weight);

            double e_risk = f_risk.extra.at(tail) + r_e + b_risk.extra.at(head);

            graph_data.weight_func->accumulate_grad(scale * ((e_risk - exp_risk) * e_marginal), *graph_data.fst, e);

            risk->accumulate_grad(scale * (e_marginal), graph, e);
        }
    }

#if 0
    frame_reconstruction_risk::frame_reconstruction_risk(
        std::vector<std::shared_ptr<autodiff::op_t>> const& frames,
        std::shared_ptr<tensor_tree::vertex> param)
        : frames(frames), param(param)
    {
        auto& g = *tensor_tree::get_var(param->children[0])->graph;

        std::shared_ptr<autodiff::op_t> input = g.var();
        std::shared_ptr<autodiff::op_t> frame = g.var();

        auto feat = autodiff::mul(input, tensor_tree::get_var(param->children[0]));
        auto diff = autodiff::sub(autodiff::add(feat, tensor_tree::get_var(param->children[1])), frame);
        auto risk = autodiff::dot(diff, diff);

        std::vector<std::shared_ptr<autodiff::op_t>> tmp_order = autodiff::topo_order(risk);

        std::unordered_set<std::shared_ptr<autodiff::op_t>> excluded {
            input, frame };

        for (int i = 0; i < 2; ++i) {
            excluded.insert(tensor_tree::get_var(param->children[i]));
        }

        std::vector<std::shared_ptr<autodiff::op_t>> order;

        for (auto& i: tmp_order) {
            if (!ebt::in(i, excluded)) {
                order.push_back(i);
            }
        }

        for (auto& i: order) {
            topo_order_shift.push_back(i->id - risk->id);
        }
    }

    double frame_reconstruction_risk::operator()(
        seg_fst<iseg_data> const& fst, int e) const
    {
        if (e < risk_cache.size() && risk_cache[e] != nullptr) {
            return autodiff::get_output<double>(risk_cache[e]);
        }

        if (e >= risk_cache.size()) {
             risk_cache.resize(e + 1, nullptr);
        }

        int tail = fst.tail(e);
        int head = fst.head(e);
        int tail_time = fst.time(tail);
        int head_time = fst.time(head);

        std::shared_ptr<composite_weight<ifst::fst>> comp_weight
            = std::static_pointer_cast<composite_weight<ifst::fst>>(fst.data.weight_func);

        std::shared_ptr<segrnn_score> weight
            = std::static_pointer_cast<segrnn_score>(comp_weight->weights[0]);

        auto feat = autodiff::mul(weight->edge_feat.at(e), tensor_tree::get_var(param->children[0]));
        auto diff = autodiff::sub(autodiff::add(feat, tensor_tree::get_var(param->children[1])),
            frames.at((tail_time + head_time) / 2));
        auto risk = autodiff::dot(diff, diff);

        risk_cache[e] = risk;

        auto& graph = *risk->graph;

        std::vector<std::shared_ptr<autodiff::op_t>> order;
        for (auto& i: topo_order_shift) {
            order.push_back(graph.vertices[risk->id + i]);
        }

        autodiff::eval(order, autodiff::eval_funcs);

        return autodiff::get_output<double>(risk_cache[e]);
    }

    void frame_reconstruction_risk::accumulate_grad(double g, seg_fst<iseg_data> const& fst, int e)
    {
        if (risk_cache[e]->grad == nullptr) {
            risk_cache[e]->grad = std::make_shared<double>(0.0);
        }

        autodiff::get_grad<double>(risk_cache[e]) += g;
    }

    void frame_reconstruction_risk::grad()
    {
        for (int e = 0; e < risk_cache.size(); ++e) {
            if (risk_cache[e] != nullptr) {
                auto& graph = *risk_cache[e]->graph;

                std::vector<std::shared_ptr<autodiff::op_t>> order;
                for (auto& i: topo_order_shift) {
                    order.push_back(graph.vertices[risk_cache[e]->id + i]);
                }
                autodiff::grad(order, autodiff::grad_funcs);
            }
        }
    }
#endif

}
