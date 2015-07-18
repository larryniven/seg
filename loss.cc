#include "scrf/loss.h"

namespace scrf {

    hinge_loss::hinge_loss(fst::path<scrf_t> const& gold, scrf_t const& graph)
        : gold(gold), graph(graph)
    {
        graph_path = shortest_path(graph, graph.topo_order);

        if (graph_path.edges().size() == 0) {
            std::cout << "no cost aug path" << std::endl;
            exit(1);
        }
    }

    real hinge_loss::loss()
    {
        real gold_score = 0;

        std::cout << "gold: ";
        for (auto& e: gold.edges()) {
            std::cout << gold.output(e) << " ";
            gold_score += gold.weight(e);
        }
        std::cout << std::endl;

        std::cout << "gold score: " << gold_score << std::endl;

        real graph_score = 0;

        std::cout << "cost aug: ";
        for (auto& e: graph_path.edges()) {
            std::cout << graph.output(e) << " ";
            graph_score += graph_path.weight(e);
        }
        std::cout << std::endl;

        std::cout << "cost aug score: " << graph_score << std::endl; 

        return graph_score - gold_score;
    }

    param_t hinge_loss::param_grad()
    {
        param_t result;

        auto const& gold_feat = *(gold.data->base_fst->feature_func);

        for (auto& e: gold.edges()) {
            param_t p;
            gold_feat(p, *(gold.data->base_fst->fst), e);

            result -= p;
        }

        // std::cout << result.class_param.at("<s>") << std::endl;

        auto const& graph_feat = *(graph.feature_func);

        for (auto& e: graph_path.edges()) {
            param_t p;
            graph_feat(p, *(graph_path.data->base_fst->fst), e);

            result += p;
        }

        // std::cout << "grad of <s>: " << result.class_param.at("<s>") << std::endl;
        // std::cout << "grad of </s>: " << result.class_param.at("</s>") << std::endl;

        return result;
    }

    filtering_loss::filtering_loss(fst::path<scrf_t> const& gold,
        scrf_t const& graph, real alpha)
        : gold(gold), graph(graph), alpha(alpha)
    {
        graph_path = shortest_path(graph, graph.topo_order);

        auto order = graph.topo_order;

        for (auto v: graph.initials()) {
            forward.extra[v] = {std::make_tuple(-1, -1), 0};
            f_param[v] = param_t {};
        }
        forward.merge(graph, order);

        for (auto& v: order) {
            auto e = forward.extra.at(v).pi;
            if (e == std::make_tuple(-1, -1)) {
                continue;
            }
            param_t p;
            (*graph.feature_func)(p, *graph.fst, e);
            p += f_param.at(graph.tail(e));
            f_param[v] = std::move(p);
        }

        std::reverse(order.begin(), order.end());

        for (auto v: graph.finals()) {
            backward.extra[v] = {std::make_tuple(-1, -1), 0};
            b_param[v] = param_t {};
        }
        backward.merge(graph, order);

        for (auto& v: order) {
            auto e = backward.extra.at(v).pi;
            if (e == std::make_tuple(-1, -1)) {
                continue;
            }
            param_t p;
            (*graph.feature_func)(p, *graph.fst, e);
            p += b_param.at(graph.head(e));
            b_param[v] = std::move(p);
        }

        auto fb_alpha = [&](std::tuple<int, int> const& v) {
            return forward.extra[v].value;
        };

        auto fb_beta = [&](std::tuple<int, int> const& v) {
            return backward.extra[v].value;
        };

        real inf = std::numeric_limits<real>::infinity();

        auto edges = graph.edges();

        real sum = 0;
        real max = -inf;

        for (auto& e: edges) {
            auto tail = graph.tail(e);
            auto head = graph.head(e);

            real s = fb_alpha(tail) + graph.weight(e) + fb_beta(head);

            if (s > max) {
                max = s;
            }

            if (s != -inf) {
                sum += s;
            }
        }

        threshold = alpha * max + (1 - alpha) * sum / edges.size();

        real f_max = -inf;

        for (auto v: graph.finals()) {
            if (forward.extra.at(v).value > f_max) {
                f_max = forward.extra.at(v).value;
            }
        }

        real b_max = -inf;

        for (auto v: graph.initials()) {
            if (backward.extra.at(v).value > b_max) {
                b_max = backward.extra.at(v).value;
            }
        }

        if (!(std::fabs(f_max - b_max) / std::fabs(b_max) < 0.001
                && std::fabs(max - b_max) / std::fabs(b_max) < 0.001)) {
            std::cout << "forward: " << f_max << " backward: " << b_max << " max: " << max << std::endl;
            exit(1);
        }
    }

    real filtering_loss::loss()
    {
        real gold_score = 0;

        for (auto e: gold.edges()) {
            gold_score += gold.weight(e);
        }

        return std::max<real>(0.0, 1 + threshold - gold_score);
    }

    param_t filtering_loss::param_grad()
    {
        param_t result;

        auto edges = graph.edges();

        for (auto e: edges) {
            param_t p;
            (*graph.feature_func)(p, *graph.fst, e);
            p += f_param.at(graph.tail(e));
            p += b_param.at(graph.head(e));

            p *= (1 - alpha) / edges.size();

            result += p;
        }

        for (auto e: graph_path.edges()) {
            param_t p;
            (*graph.feature_func)(p, *(graph_path.data->base_fst->fst), e);

            p *= alpha;

            result += p;
        }

        for (auto e: gold.edges()) {
            param_t p;
            (*gold.data->base_fst->feature_func)(p, *(gold.data->base_fst->fst), e);
            result -= p;
        }

        return result;
    }

    hinge_loss_beam::hinge_loss_beam(fst::path<scrf_t> const& gold, scrf_t const& graph,
        int beam_width)
        : gold(gold), graph(graph), beam_width(beam_width)
    {
        fst::beam_search<scrf_t> beam_search;
        beam_search.search(graph, beam_width);
        graph_path = beam_search.best_path(graph);

        if (graph_path.edges().size() == 0) {
            std::cout << "no cost aug path" << std::endl;
            exit(1);
        }
    }

    real hinge_loss_beam::loss()
    {
        real gold_score = 0;

        std::cout << "gold: ";
        for (auto& e: gold.edges()) {
            std::cout << gold.output(e) << " ";
            gold_score += gold.weight(e);
        }
        std::cout << std::endl;

        std::cout << "gold score: " << gold_score << std::endl;

        real graph_score = 0;

        std::cout << "cost aug: ";
        for (auto& e: graph_path.edges()) {
            std::cout << graph.output(e) << " ";
            graph_score += graph_path.weight(e);
        }
        std::cout << std::endl;

        std::cout << "cost aug score: " << graph_score << std::endl; 

        return graph_score - gold_score;
    }

    param_t hinge_loss_beam::param_grad()
    {
        param_t result;

        auto const& gold_feat = *(gold.data->base_fst->feature_func);

        for (auto& e: gold.edges()) {
            param_t p;
            gold_feat(p, *(gold.data->base_fst->fst), e);

            result -= p;
        }

        auto const& graph_feat = *(graph.feature_func);

        for (auto& e: graph_path.edges()) {
            param_t p;
            graph_feat(p, *(graph_path.data->base_fst->fst), e);

            result += p;
        }

        return result;
    }

}
