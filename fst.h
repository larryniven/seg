#ifndef FST_H
#define FST_H

#include "scrf/util.h"
#include <unordered_map>
#include <tuple>
#include <vector>
#include <limits>
#include <algorithm>
#include <memory>
#include "ebt/ebt.h"

namespace fst {

    template <class fst>
    struct sub_fst_data {
        using vertex_type = typename fst::vertex_type;
        using edge_type = typename fst::edge_type;

        fst const* base_fst;
        std::vector<vertex_type> vertices;
        std::vector<edge_type> edges;
        std::unordered_map<vertex_type, std::vector<edge_type>> in_edges;
        std::unordered_map<vertex_type, std::vector<edge_type>> out_edges;

        std::vector<vertex_type> initials;
        std::vector<vertex_type> finals;
    };

    template <class fst>
    struct sub_fst {
        using vertex_type = typename fst::vertex_type;
        using edge_type = typename fst::edge_type;

        std::shared_ptr<sub_fst_data<fst>> data;

        std::vector<vertex_type> vertices() const
        {
            return data->vertices;
        }

        std::vector<edge_type> edges() const
        {
            return data->edges;
        }

        real weight(edge_type const& e) const
        {
            return data->base_fst->weight(e);
        }

        vertex_type tail(edge_type const& e) const
        {
            return data->base_fst->tail(e);
        }

        vertex_type head(edge_type const& e) const
        {
            return data->base_fst->head(e);
        }

        std::vector<edge_type> in_edges(vertex_type const& v) const
        {
            return data->in_edges.at(v);
        }

        std::vector<edge_type> out_edges(vertex_type const& v) const
        {
            return data->out_edges.at(v);
        }

        std::vector<vertex_type> initials() const
        {
            return data->initials;
        }

        std::vector<vertex_type> finals() const
        {
            return data->finals;
        }

        std::string input(edge_type const& e) const
        {
            return data->base_fst->input(e);
        }

        std::string output(edge_type const& e) const
        {
            return data->base_fst->output(e);
        }

    };

    template <class fst> using path_data = sub_fst_data<fst>;
    template <class fst> using path = sub_fst<fst>;

    template <class fst>
    sub_fst<fst> make_path(fst const& f, std::vector<typename fst::edge_type> const& edges)
    {
        sub_fst_data<fst> result_data;

        result_data.base_fst = &f;
        result_data.edges = edges;

        std::unordered_set<typename fst::vertex_type> vertices;

        typename fst::vertex_type initial = f.tail(edges.front());
        typename fst::vertex_type final = f.head(edges.back());

        for (auto& e: edges) {
            vertices.insert(f.tail(e));
            vertices.insert(f.head(e));

            result_data.in_edges[f.head(e)].push_back(e);
            result_data.out_edges[f.tail(e)].push_back(e);
        }

        result_data.vertices = std::vector<typename fst::vertex_type> {
            vertices.begin(), vertices.end() };

        result_data.initials.push_back(initial);
        result_data.finals.push_back(final);

        sub_fst<fst> result;

        result.data = std::make_shared<sub_fst_data<fst>>(std::move(result_data));

        return result;
    }

    template <class... Args>
    struct composed_fst;

    template <class fst_type1, class fst_type2>
    struct composed_fst<fst_type1, fst_type2> {
        using edge_type = std::tuple<typename fst_type1::edge_type,
            typename fst_type2::edge_type>;

        using vertex_type = std::tuple<typename fst_type1::vertex_type,
            typename fst_type2::vertex_type>;

        std::shared_ptr<fst_type1> fst1;
        std::shared_ptr<fst_type2> fst2;

        std::vector<vertex_type> vertices() const
        {
            std::vector<vertex_type> result;

            for (auto& v1: fst1->vertices()) {
                for (auto& v2: fst2->vertices()) {
                    result.push_back(std::make_tuple(v1, v2));
                }
            }

            return result;
        }

        std::vector<edge_type> edges() const
        {
            std::vector<edge_type> result;

            for (auto& e1: fst1->edges()) {
                for (auto& e2: fst2->edges()) {
                    if (fst1->output(e1) == fst2->input(e2)) {
                        result.push_back(std::make_tuple(e1, e2));
                    }
                }
            }

            return result;
        }

        std::vector<edge_type> in_edges(vertex_type const& v) const
        {
            std::vector<edge_type> result;

            auto& e2_set = fst2->in_edges_map(std::get<1>(v));

            for (auto& e1: fst1->in_edges(std::get<0>(v))) {
                if (!ebt::in(fst1->output(e1), e2_set)) {
                    continue;
                }

                for (auto& e2: e2_set.at(fst1->output(e1))) {
                    result.push_back(std::make_tuple(e1, e2));
                }
            }

            return result;
        }

        std::vector<edge_type> out_edges(vertex_type const& v) const
        {
            std::vector<edge_type> result;

            auto& e2_set = fst2->out_edges_map(std::get<1>(v));

            for (auto& e1: fst1->out_edges(std::get<0>(v))) {
                if (!ebt::in(fst1->output(e1), e2_set)) {
                    continue;
                }

                for (auto& e2: e2_set.at(fst1->output(e1))) {
                    result.push_back(std::make_tuple(e1, e2));
                }
            }

            return result;
        }

        real weight(edge_type const& e) const
        {
            return fst1->weight(std::get<0>(e)) + fst2->weight(std::get<1>(e));
        }

        vertex_type tail(edge_type const& e) const
        {
            return std::make_tuple(fst1->tail(std::get<0>(e)),
                fst2->tail(std::get<1>(e)));
        }

        vertex_type head(edge_type const& e) const
        {
            return std::make_tuple(fst1->head(std::get<0>(e)),
                fst2->head(std::get<1>(e)));
        }

        std::vector<vertex_type> initials() const
        {
            std::vector<vertex_type> result;

            for (auto i1: fst1->initials()) {
                for (auto i2: fst2->initials()) {
                    result.push_back(std::make_tuple(i1, i2));
                }
            }

            return result;
        }

        std::vector<vertex_type> finals() const
        {
            std::vector<vertex_type> result;

            for (auto i1: fst1->finals()) {
                for (auto i2: fst2->finals()) {
                    result.push_back(std::make_tuple(i1, i2));
                }
            }

            return result;
        }

        std::string input(edge_type const& e) const
        {
            return fst1->input(std::get<0>(e));
        }

        std::string output(edge_type const& e) const
        {
            return fst2->output(std::get<1>(e));
        }

    };

    template <class fst_type1, class fst_type2, class fst_type3>
    struct composed_fst<fst_type1, fst_type2, fst_type3> {
        using edge_type = std::tuple<typename fst_type1::edge_type,
            typename fst_type2::edge_type, typename fst_type3::edge_type>;
        using vertex_type = std::tuple<typename fst_type1::vertex_type,
            typename fst_type2::vertex_type, typename fst_type3::vertex_type>;

        std::shared_ptr<fst_type1> fst1;
        std::shared_ptr<fst_type2> fst2;
        std::shared_ptr<fst_type3> fst3;

        std::vector<vertex_type> vertices() const
        {
            std::vector<vertex_type> result;

            for (auto& v1: fst1->vertices()) {
                for (auto& v2: fst2->vertices()) {
                    for (auto& v3: fst3->vertices()) {
                        result.push_back(std::make_tuple(v1, v2, v3));
                    }
                }
            }

            return result;
        }

        std::vector<edge_type> edges() const
        {
            std::vector<edge_type> result;

            for (auto& e1: fst1->edges()) {
                for (auto& e2: fst2->edges()) {
                    for (auto& e3: fst3->edges()) {
                        if (fst1->output(e1) == fst2->input(e2) &&
                                fst2->output(e2) == fst3->input(e3)) {
                            result.push_back(std::make_tuple(e1, e2, e3));
                        }
                    }
                }
            }

            return result;
        }

        std::vector<edge_type> in_edges(vertex_type const& v) const
        {
            std::vector<edge_type> result;

            auto& e2_set = fst2->in_edges_map(std::get<1>(v));
            auto& e3_set = fst3->in_edges_map(std::get<2>(v));

            for (auto& e1: fst1->in_edges(std::get<0>(v))) {
                if (!ebt::in(fst1->output(e1), e2_set)) {
                    continue;
                }

                for (auto& e2: e2_set.at(fst1->output(e1))) {
                    if (!ebt::in(fst2->output(e2), e3_set)) {
                        continue;
                    }

                    for (auto& e3: e3_set.at(fst2->output(e2))) {
                        result.push_back(std::make_tuple(e1, e2, e3));
                    }
                }
            }

            return result;
        }

        std::vector<edge_type> out_edges(vertex_type const& v) const
        {
            std::vector<edge_type> result;

            auto& e2_set = fst2->out_edges_map(std::get<1>(v));
            auto& e3_set = fst3->out_edges_map(std::get<2>(v));

            for (auto& e1: fst1->out_edges(std::get<0>(v))) {
                if (!ebt::in(fst1->output(e1), e2_set)) {
                    continue;
                }

                for (auto& e2: e2_set.at(fst1->output(e1))) {
                    if (!ebt::in(fst2->output(e2), e3_set)) {
                        continue;
                    }

                    for (auto& e3: e3_set.at(fst2->output(e2))) {
                        result.push_back(std::make_tuple(e1, e2, e3));
                    }
                }
            }

            return result;
        }

        real weight(edge_type const& e) const
        {
            return fst1->weight(std::get<0>(e)) + fst2->weight(std::get<1>(e))
                + fst3->weight(std::get<2>(e));
        }

        vertex_type tail(edge_type const& e) const
        {
            return std::make_tuple(fst1->tail(std::get<0>(e)),
                fst2->tail(std::get<1>(e)),
                fst3->tail(std::get<2>(e)));
        }

        vertex_type head(edge_type const& e) const
        {
            return std::make_tuple(fst1->head(std::get<0>(e)),
                fst2->head(std::get<1>(e)),
                fst3->head(std::get<2>(e)));
        }

        std::vector<vertex_type> initials() const
        {
            std::vector<vertex_type> result;

            for (auto i1: fst1->initials()) {
                for (auto i2: fst2->initials()) {
                    for (auto i3: fst3->initials()) {
                        result.push_back(std::make_tuple(i1, i2, i3));
                    }
                }
            }

            return result;
        }

        std::vector<vertex_type> finals() const
        {
            std::vector<vertex_type> result;

            for (auto i1: fst1->finals()) {
                for (auto i2: fst2->finals()) {
                    for (auto i3: fst3->finals()) {
                        result.push_back(std::make_tuple(i1, i2, i3));
                    }
                }
            }

            return result;
        }

        std::string input(edge_type const& e) const
        {
            return fst1->input(std::get<0>(e));
        }

        std::string output(edge_type const& e) const
        {
            return fst3->output(std::get<2>(e));
        }

    };

    template <class fst_type>
    struct beam_search {
        using vertex_type = typename fst_type::vertex_type;
        using edge_type = typename fst_type::edge_type;

        std::unordered_map<vertex_type, real> score;
        std::unordered_map<vertex_type, edge_type> pi;

        void search(fst_type const& fst, int top_k)
        {
            ebt::MaxHeap<vertex_type, real> to_expand;
            std::unordered_set<vertex_type> to_expand_set;

            for (auto& i: fst.initials()) {
                to_expand.insert(i, 0);
                score[i] = 0;
            }

            real inf = std::numeric_limits<real>::infinity();

            while (to_expand.size() > 0) {
                ebt::MaxHeap<vertex_type, real> expanded;
                std::unordered_set<vertex_type> expanded_set;

                while (to_expand.size()) {
                    auto v = to_expand.extract_max();
                    to_expand_set.erase(v);

                    for (auto& e: fst.out_edges(v)) {
                        vertex_type const& head = fst.head(e);
                        real s = fst.weight(e) + score.at(v);
                        if (s > ebt::get(score, head, -inf)) {
                            score[head] = s;
                            pi[head] = e;
                        }

                        if (ebt::in(head, expanded_set)) {
                            if (s > ebt::get(score, head, -inf)) {
                                expanded.increase_key(head, s);
                            }
                        } else {
                            expanded.insert(head, score.at(head));
                            expanded_set.insert(head);
                        }
                    }
                }

                for (int i = 0; i < top_k && expanded.size() > 0; ++i) {
                    auto v = expanded.extract_max();
                    if (ebt::in(v, to_expand_set)) {
                        to_expand.increase_key(v, score.at(v));
                    } else {
                        to_expand.insert(v, score.at(v));
                        to_expand_set.insert(v);
                    }
                }
            }
        }

        path<fst_type> best_path(fst_type const& fst)
        {
            path_data<fst_type> result { &fst };

            double inf = std::numeric_limits<double>::infinity();
            double max = -inf;
            vertex_type argmax;
            for (auto& f: fst.finals()) {
                if (ebt::get(score, f, -inf) > max) {
                    max = ebt::get(score, f, -inf);
                    argmax = f;
                }
            }
            result.finals.push_back(argmax);
            result.vertices.push_back(argmax);

            auto add_edge = [&](edge_type const& e) {
                result.vertices.push_back(fst.tail(e));
                result.edges.push_back(e);
                result.in_edges[fst.head(e)].push_back(e);
                result.out_edges[fst.tail(e)].push_back(e);
            };

            auto u = argmax;
            while (ebt::in(u, pi)) {
                add_edge(pi.at(u));
                u = fst.tail(pi.at(u));
            }

            result.initials.push_back(u);

            std::reverse(result.vertices.begin(), result.vertices.end());
            std::reverse(result.edges.begin(), result.edges.end());

            path<fst_type> p;
            p.data = std::make_shared<path_data<fst_type>>(result);

            return p;
        }
    };

    template <class fst_type>
    struct forward_one_best {

        using vertex_type = typename fst_type::vertex_type;
        using edge_type = typename fst_type::edge_type;

        struct extra_data {
            edge_type pi;
            real value;
        };

        std::unordered_map<vertex_type, extra_data> extra;

        void merge(fst_type const& fst, std::vector<vertex_type> const& order)
        {
            auto get_value = [&](vertex_type v) {
                real inf = std::numeric_limits<real>::infinity();

                if (!ebt::in(v, extra)) {
                    return -inf;
                } else {
                    return extra.at(v).value;
                }
            };

            for (auto& u: order) {
                real max = get_value(u);
                edge_type argmax;
                bool update = false;

                for (auto&& e: fst.in_edges(u)) {
                    vertex_type v = fst.tail(e);

                    real candidate_value = get_value(v) + fst.weight(e);

                    if (candidate_value > max) {
                        max = candidate_value;
                        argmax = e;
                        update = true;
                    }
                }

                if (update) {
                    extra[u] = extra_data { argmax, max };
                }
            }
        }

        path<fst_type> best_path(fst_type const& fst)
        {
            real inf = std::numeric_limits<real>::infinity();
            real max = -inf;
            vertex_type argmax;

            for (auto v: fst.finals()) {
                if (ebt::in(v, extra) && extra.at(v).value > max) {
                    max = extra.at(v).value;
                    argmax = v;
                }
            }

            path_data<fst_type> result;
            result.base_fst = &fst;

            if (max == -inf) {
                path<fst_type> p;
                p.data = std::make_shared<path_data<fst_type>>(result);
                return p;
            }

            vertex_type u = argmax;

            result.vertices.push_back(u);

            auto initials = fst.initials();
            std::unordered_set<vertex_type> initial_set { initials.begin(), initials.end() };

            while (!ebt::in(u, initial_set)) {
                edge_type e = extra.at(u).pi;

                vertex_type v = fst.tail(e);

                result.edges.push_back(e);
                result.in_edges[u].push_back(e);
                result.out_edges[v].push_back(e);

                result.vertices.push_back(v);

                u = v;
            }

            result.initials.push_back(u);
            result.finals.push_back(argmax);

            std::reverse(result.vertices.begin(), result.vertices.end());
            std::reverse(result.edges.begin(), result.edges.end());

            path<fst_type> p;
            p.data = std::make_shared<path_data<fst_type>>(std::move(result));

            return p;
        }

    };

    template <class fst_type>
    using one_best = forward_one_best<fst_type>;

    template <class fst_type>
    struct backward_one_best {

        using vertex_type = typename fst_type::vertex_type;
        using edge_type = typename fst_type::edge_type;

        struct extra_data {
            edge_type pi;
            real value;
        };

        std::unordered_map<vertex_type, extra_data> extra;

        void merge(fst_type const& fst, std::vector<vertex_type> const& order)
        {
            auto get_value = [&](vertex_type v) {
                real inf = std::numeric_limits<real>::infinity();

                if (!ebt::in(v, extra)) {
                    return -inf;
                } else {
                    return extra.at(v).value;
                }
            };

            for (auto& u: order) {
                real max = get_value(u);
                edge_type argmax;
                bool update = false;

                for (auto&& e: fst.out_edges(u)) {
                    vertex_type v = fst.head(e);

                    real candidate_value = get_value(v) + fst.weight(e);
                    if (candidate_value > max) {
                        max = candidate_value;
                        argmax = e;
                        update = true;
                    }
                }

                if (update) {
                    extra[u] = extra_data { argmax, max };
                }
            }
        }

        path<fst_type> best_path(fst_type const& fst)
        {
            real inf = std::numeric_limits<real>::infinity();
            real max = -inf;
            vertex_type argmax;

            for (auto v: fst.initials()) {
                if (ebt::in(v, extra) && extra.at(v).value > max) {
                    max = extra.at(v).value;
                    argmax = v;
                }
            }

            path_data<fst_type> result;
            result.base_fst = &fst;

            if (max == -inf) {
                path<fst_type> p;
                p.data = std::make_shared<path_data<fst_type>>(result);
                return p;
            }

            vertex_type u = argmax;
            result.vertices.push_back(u);

            auto finals = fst.finals();
            std::unordered_set<vertex_type> final_set { finals.begin(), finals.end() };

            while (!ebt::in(u, final_set)) {
                edge_type e = extra.at(u).pi;

                vertex_type v = fst.head(e);

                result.edges.push_back(e);
                result.in_edges[v].push_back(e);
                result.out_edges[u].push_back(e);

                result.vertices.push_back(v);

                u = v;
            }

            result.initials.push_back(argmax);
            result.finals.push_back(u);

            path<fst_type> p;
            p.data = std::make_shared<path_data<fst_type>>(std::move(result));

            return p;
        }

    };

    template <class fst_type>
    struct lazy_k_best {
        struct card_t {
            double value;
            typename fst_type::edge_type edge;
            int index;
        };

        std::unordered_map<typename fst_type::vertex_type, std::vector<card_t>> deck;

        std::unordered_map<typename fst_type::edge_type, int> tail_index;

        void one_best(fst_type const& fst,
            std::vector<typename fst_type::vertex_type> const& topo_order)
        {
            double inf = std::numeric_limits<double>::infinity();

            for (auto& v: topo_order) {
                auto edges = fst.in_edges(v);

                if (edges.size() == 0) {
                    deck[v].push_back(card_t {0, typename fst_type::edge_type {}, -1});
                } else {
                    for (auto& e: edges) {
                        tail_index[e] = 0;
                    }

                    double max = -inf;
                    typename fst_type::edge_type argmax;

                    for (auto& e: edges) {
                        double score = deck.at(fst.tail(e)).at(0).value + fst.weight(e);

                        if (score > max) {
                            max = score;
                            argmax = e;
                        }
                    }

                    deck[v].push_back(card_t {max, argmax, 0});
                }
            }
        }

        path<fst_type> backtrack(fst_type const& fst,
            typename fst_type::vertex_type const& v, int index)
        {
            std::vector<typename fst_type::edge_type> edges;

            auto u = v;
            int i = index;

            while (1) {
                card_t c = deck.at(u).at(i);

                if (c.index == -1) {
                    break;
                }

                edges.push_back(c.edge);

                typename fst_type::edge_type e = c.edge;
                u = fst.tail(e);
                i = c.index;
            }

            std::reverse(edges.begin(), edges.end());

            return make_path(fst, edges);
        }

        void update(fst_type const& fst, path<fst_type> const& path)
        {
            double inf = std::numeric_limits<double>::infinity();

            for (auto& e: path.edges()) {
                if (tail_index.at(e) == -1) {
                    continue;
                }

                if (tail_index.at(e) < deck.at(fst.tail(e)).size() - 1) {
                    tail_index.at(e) += 1;
                } else {
                    tail_index.at(e) = -1;
                }

                auto edges = fst.in_edges(fst.head(e));

                double max = -inf;
                typename fst_type::edge_type argmax;

                for (auto& e2: edges) {
                    if (tail_index.at(e2) == -1) {
                        continue;
                    }

                    double score = deck.at(fst.tail(e2)).at(tail_index.at(e2)).value + fst.weight(e2);

                    if (score > max) {
                        max = score;
                        argmax = e2;
                    }
                }

                if (max != -inf) {
                    deck[fst.head(e)].push_back(card_t {max, argmax, tail_index.at(argmax)});
                }
            }
        }
    };

    template <class fst>
    std::vector<typename fst::vertex_type> topo_order(fst const& f)
    {
        enum class action_t {
            color_grey,
            color_black
        };

        std::vector<std::pair<action_t, typename fst::vertex_type>> stack;
        std::unordered_set<typename fst::vertex_type> traversed;

        std::vector<typename fst::vertex_type> order;

        for (auto& v: f.initials()) {
            stack.push_back(std::make_pair(action_t::color_grey, v));
        }

        while (stack.size() > 0) {
            action_t a;
            typename fst::vertex_type v;
            std::tie(a, v) = stack.back();
            stack.pop_back();

            if (a == action_t::color_grey) {
                if (ebt::in(v, traversed)) {
                    continue;
                }

                traversed.insert(v);

                stack.push_back(std::make_pair(action_t::color_black, v));

                for (auto& e: f.out_edges(v)) {
                    auto u = f.head(e);

                    if (!ebt::in(u, traversed)) {
                        stack.push_back(std::make_pair(action_t::color_grey, u));
                    }
                }

            } else if (a == action_t::color_black) {
                order.push_back(v);
            } else {
                std::cerr << "unknown action " << int(a) << std::endl;
                exit(1);
            }
        }

        std::reverse(order.begin(), order.end());

        return order;
    }

}

#endif
