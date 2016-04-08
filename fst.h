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
        using vertex = typename fst::vertex;
        using edge = typename fst::edge;

        fst const* base_fst;
        std::vector<vertex> vertices;
        std::vector<edge> edges;
        std::unordered_map<vertex, std::vector<edge>> in_edges;
        std::unordered_map<vertex, std::vector<edge>> out_edges;

        std::vector<vertex> initials;
        std::vector<vertex> finals;
    };

    template <class fst>
    struct sub_fst {
        using vertex = typename fst::vertex;
        using edge = typename fst::edge;

        std::shared_ptr<sub_fst_data<fst>> data;

        std::vector<vertex> vertices() const
        {
            return data->vertices;
        }

        std::vector<edge> edges() const
        {
            return data->edges;
        }

        real weight(edge const& e) const
        {
            return data->base_fst->weight(e);
        }

        vertex tail(edge const& e) const
        {
            return data->base_fst->tail(e);
        }

        vertex head(edge const& e) const
        {
            return data->base_fst->head(e);
        }

        std::vector<edge> in_edges(vertex const& v) const
        {
            return data->in_edges.at(v);
        }

        std::vector<edge> out_edges(vertex const& v) const
        {
            return data->out_edges.at(v);
        }

        std::vector<vertex> initials() const
        {
            return data->initials;
        }

        std::vector<vertex> finals() const
        {
            return data->finals;
        }

        auto input(edge const& e) const -> decltype(data->base_fst->input(e))
        {
            return data->base_fst->input(e);
        }

        auto output(edge const& e) const -> decltype(data->base_fst->output(e))
        {
            return data->base_fst->output(e);
        }

    };

    template <class fst> using path_data = sub_fst_data<fst>;
    template <class fst> using path = sub_fst<fst>;

    template <class fst>
    sub_fst<fst> make_path(fst const& f, std::vector<typename fst::edge> const& edges)
    {
        sub_fst_data<fst> result_data;

        result_data.base_fst = &f;
        result_data.edges = edges;

        std::unordered_set<typename fst::vertex> vertices;

        typename fst::vertex initial = f.tail(edges.front());
        typename fst::vertex final = f.head(edges.back());

        for (auto& e: edges) {
            vertices.insert(f.tail(e));
            vertices.insert(f.head(e));

            result_data.in_edges[f.head(e)].push_back(e);
            result_data.out_edges[f.tail(e)].push_back(e);
        }

        result_data.vertices = std::vector<typename fst::vertex> {
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
        using edge = std::tuple<typename fst_type1::edge,
            typename fst_type2::edge>;

        using vertex = std::tuple<typename fst_type1::vertex,
            typename fst_type2::vertex>;

        std::shared_ptr<fst_type1> fst1;
        std::shared_ptr<fst_type2> fst2;

        std::vector<vertex> vertices() const
        {
            std::vector<vertex> result;

            for (auto& v1: fst1->vertices()) {
                for (auto& v2: fst2->vertices()) {
                    result.push_back(std::make_tuple(v1, v2));
                }
            }

            return result;
        }

        std::vector<edge> edges() const
        {
            std::vector<edge> result;

            for (auto& e1: fst1->edges()) {
                for (auto& e2: fst2->edges()) {
                    if (fst1->output(e1) == fst2->input(e2)) {
                        result.push_back(std::make_tuple(e1, e2));
                    }
                }
            }

            return result;
        }

        std::vector<edge> in_edges(vertex const& v) const
        {
            std::vector<edge> result;

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

        std::vector<edge> out_edges(vertex const& v) const
        {
            std::vector<edge> result;

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

        real weight(edge const& e) const
        {
            return fst1->weight(std::get<0>(e)) + fst2->weight(std::get<1>(e));
        }

        vertex tail(edge const& e) const
        {
            return std::make_tuple(fst1->tail(std::get<0>(e)),
                fst2->tail(std::get<1>(e)));
        }

        vertex head(edge const& e) const
        {
            return std::make_tuple(fst1->head(std::get<0>(e)),
                fst2->head(std::get<1>(e)));
        }

        std::vector<vertex> initials() const
        {
            std::vector<vertex> result;

            for (auto i1: fst1->initials()) {
                for (auto i2: fst2->initials()) {
                    result.push_back(std::make_tuple(i1, i2));
                }
            }

            return result;
        }

        std::vector<vertex> finals() const
        {
            std::vector<vertex> result;

            for (auto i1: fst1->finals()) {
                for (auto i2: fst2->finals()) {
                    result.push_back(std::make_tuple(i1, i2));
                }
            }

            return result;
        }

        std::string input(edge const& e) const
        {
            return fst1->input(std::get<0>(e));
        }

        std::string output(edge const& e) const
        {
            return fst2->output(std::get<1>(e));
        }

    };

    template <class fst_type1, class fst_type2, class fst_type3>
    struct composed_fst<fst_type1, fst_type2, fst_type3> {
        using edge = std::tuple<typename fst_type1::edge,
            typename fst_type2::edge, typename fst_type3::edge>;
        using vertex = std::tuple<typename fst_type1::vertex,
            typename fst_type2::vertex, typename fst_type3::vertex>;

        std::shared_ptr<fst_type1> fst1;
        std::shared_ptr<fst_type2> fst2;
        std::shared_ptr<fst_type3> fst3;

        std::vector<vertex> vertices() const
        {
            std::vector<vertex> result;

            for (auto& v1: fst1->vertices()) {
                for (auto& v2: fst2->vertices()) {
                    for (auto& v3: fst3->vertices()) {
                        result.push_back(std::make_tuple(v1, v2, v3));
                    }
                }
            }

            return result;
        }

        std::vector<edge> edges() const
        {
            std::vector<edge> result;

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

        std::vector<edge> in_edges(vertex const& v) const
        {
            std::vector<edge> result;

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

        std::vector<edge> out_edges(vertex const& v) const
        {
            std::vector<edge> result;

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

        real weight(edge const& e) const
        {
            return fst1->weight(std::get<0>(e)) + fst2->weight(std::get<1>(e))
                + fst3->weight(std::get<2>(e));
        }

        vertex tail(edge const& e) const
        {
            return std::make_tuple(fst1->tail(std::get<0>(e)),
                fst2->tail(std::get<1>(e)),
                fst3->tail(std::get<2>(e)));
        }

        vertex head(edge const& e) const
        {
            return std::make_tuple(fst1->head(std::get<0>(e)),
                fst2->head(std::get<1>(e)),
                fst3->head(std::get<2>(e)));
        }

        std::vector<vertex> initials() const
        {
            std::vector<vertex> result;

            for (auto i1: fst1->initials()) {
                for (auto i2: fst2->initials()) {
                    for (auto i3: fst3->initials()) {
                        result.push_back(std::make_tuple(i1, i2, i3));
                    }
                }
            }

            return result;
        }

        std::vector<vertex> finals() const
        {
            std::vector<vertex> result;

            for (auto i1: fst1->finals()) {
                for (auto i2: fst2->finals()) {
                    for (auto i3: fst3->finals()) {
                        result.push_back(std::make_tuple(i1, i2, i3));
                    }
                }
            }

            return result;
        }

        std::string input(edge const& e) const
        {
            return fst1->input(std::get<0>(e));
        }

        std::string output(edge const& e) const
        {
            return fst3->output(std::get<2>(e));
        }

    };

    template <class fst_type>
    struct beam_search {
        using vertex = typename fst_type::vertex;
        using edge = typename fst_type::edge;

        std::unordered_map<vertex, real> score;
        std::unordered_map<vertex, edge> pi;

        void search(fst_type const& fst, int top_k)
        {
            ebt::MaxHeap<vertex, real> to_expand;
            std::unordered_set<vertex> to_expand_set;

            for (auto& i: fst.initials()) {
                to_expand.insert(i, 0);
                score[i] = 0;
            }

            real inf = std::numeric_limits<real>::infinity();

            while (to_expand.size() > 0) {
                ebt::MaxHeap<vertex, real> expanded;
                std::unordered_set<vertex> expanded_set;

                while (to_expand.size()) {
                    auto v = to_expand.extract_max();
                    to_expand_set.erase(v);

                    for (auto& e: fst.out_edges(v)) {
                        vertex const& head = fst.head(e);
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
            vertex argmax;
            for (auto& f: fst.finals()) {
                if (ebt::get(score, f, -inf) > max) {
                    max = ebt::get(score, f, -inf);
                    argmax = f;
                }
            }
            result.finals.push_back(argmax);
            result.vertices.push_back(argmax);

            auto add_edge = [&](edge const& e) {
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

        using vertex = typename fst_type::vertex;
        using edge = typename fst_type::edge;

        struct extra_data {
            edge pi;
            real value;
        };

        std::unordered_map<vertex, extra_data> extra;

        void merge(fst_type const& fst, std::vector<vertex> const& order)
        {
            auto get_value = [&](vertex v) {
                real inf = std::numeric_limits<real>::infinity();

                if (!ebt::in(v, extra)) {
                    return -inf;
                } else {
                    return extra.at(v).value;
                }
            };

            for (auto& u: order) {
                real max = get_value(u);
                edge argmax;
                bool update = false;

                std::vector<edge> edges = fst.in_edges(u);
                std::vector<double> candidate_value;
                candidate_value.resize(edges.size());

                #pragma omp parallel for
                for (int i = 0; i < edges.size(); ++i) {
                    edge& e = edges[i];
                    vertex v = fst.tail(e);
                    candidate_value[i] = get_value(v) + fst.weight(e);
                }

                for (int i = 0; i < edges.size(); ++i) {
                    if (candidate_value[i] > max) {
                        max = candidate_value[i];
                        argmax = edges[i];
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
            vertex argmax;

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

            vertex u = argmax;

            result.vertices.push_back(u);

            auto initials = fst.initials();
            std::unordered_set<vertex> initial_set { initials.begin(), initials.end() };

            while (!ebt::in(u, initial_set)) {
                edge e = extra.at(u).pi;

                vertex v = fst.tail(e);

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

        using vertex = typename fst_type::vertex;
        using edge = typename fst_type::edge;

        struct extra_data {
            edge pi;
            real value;
        };

        std::unordered_map<vertex, extra_data> extra;

        void merge(fst_type const& fst, std::vector<vertex> const& order)
        {
            auto get_value = [&](vertex v) {
                real inf = std::numeric_limits<real>::infinity();

                if (!ebt::in(v, extra)) {
                    return -inf;
                } else {
                    return extra.at(v).value;
                }
            };

            for (auto& u: order) {
                real max = get_value(u);
                edge argmax;
                bool update = false;

                for (auto&& e: fst.out_edges(u)) {
                    vertex v = fst.head(e);

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
            vertex argmax;

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

            vertex u = argmax;
            result.vertices.push_back(u);

            auto finals = fst.finals();
            std::unordered_set<vertex> final_set { finals.begin(), finals.end() };

            while (!ebt::in(u, final_set)) {
                edge e = extra.at(u).pi;

                vertex v = fst.head(e);

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
            typename fst_type::edge edge;
            int index;
        };

        std::unordered_map<typename fst_type::vertex, std::vector<card_t>> deck;

        std::unordered_map<typename fst_type::edge, int> tail_index;

        void one_best(fst_type const& fst,
            std::vector<typename fst_type::vertex> const& topo_order)
        {
            double inf = std::numeric_limits<double>::infinity();

            for (auto& v: topo_order) {
                auto edges = fst.in_edges(v);

                if (edges.size() == 0) {
                    deck[v].push_back(card_t {0, typename fst_type::edge {}, -1});
                } else {
                    for (auto& e: edges) {
                        tail_index[e] = 0;
                    }

                    double max = -inf;
                    typename fst_type::edge argmax;

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
            typename fst_type::vertex const& v, int index)
        {
            std::vector<typename fst_type::edge> edges;

            auto u = v;
            int i = index;

            while (1) {
                card_t c = deck.at(u).at(i);

                if (c.index == -1) {
                    break;
                }

                edges.push_back(c.edge);

                typename fst_type::edge e = c.edge;
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
                typename fst_type::edge argmax;

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
    std::vector<typename fst::vertex> topo_order(fst const& f)
    {
        enum class action_t {
            color_grey,
            color_black
        };

        std::vector<std::pair<action_t, typename fst::vertex>> stack;
        std::unordered_set<typename fst::vertex> traversed;

        std::vector<typename fst::vertex> order;

        for (auto& v: f.initials()) {
            stack.push_back(std::make_pair(action_t::color_grey, v));
        }

        while (stack.size() > 0) {
            action_t a;
            typename fst::vertex v;
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

    namespace experimental {

        template <class vertex, class edge>
        struct graph {
            virtual ~graph()
            {}

            virtual std::vector<vertex> const& vertices() const = 0;
            virtual std::vector<vertex> const& edges() const = 0;
            virtual vertex tail(edge e) const = 0;
            virtual vertex head(edge e) const = 0;
            virtual std::vector<edge> const& in_edges(vertex v) const = 0;
            virtual std::vector<edge> const& out_edges(vertex v) const = 0;
            virtual std::vector<vertex> const& initials() const = 0;
            virtual std::vector<vertex> const& finals() const = 0;
        };

        template <class edge>
        struct weighted {
            virtual double weight(edge e) const = 0;
        };

        template <class vertex>
        struct with_topo_order {
            virtual std::vector<vertex> const& topo_order() const = 0;
        };

        template <class vertex, class edge, class symbol>
        struct fst
            : public graph<vertex, edge>
            , weighted<edge> {

            virtual symbol const& input(edge e) const = 0;
            virtual symbol const& output(edge e) const = 0;
        };

        template <class vertex>
        struct timed {
            virtual long time(vertex v) const = 0;
        };

        template <class vertex, class edge, class symbol>
        struct adj_indexed {
            virtual std::unordered_map<symbol, std::vector<edge>> const&
            in_edges_map(vertex v) const = 0;

            virtual std::unordered_map<symbol, std::vector<edge>> const&
            out_edges_map(vertex v) const = 0;
        };

        template <class symbol>
        struct symbol_trait;

        template <>
        struct symbol_trait<std::string> {
            static std::string eps;
        };

        template <>
        struct symbol_trait<int> {
            static int eps;
        };

        template <class edge>
        struct edge_trait;

        template <>
        struct edge_trait<int> {
            static int null;
        };

        template <class fst>
        struct path_maker {
            virtual fst operator()(std::vector<typename fst::edge> const& edges, fst const& f) const = 0;
        };

        template <class fst>
        struct forward_one_best {

            using vertex = typename fst::vertex;
            using edge = typename fst::edge;

            struct extra_data {
                edge pi;
                double value;
            };

            std::unordered_map<vertex, extra_data> extra;

            void merge(fst const& f, std::vector<vertex> const& order)
            {
                double inf = std::numeric_limits<real>::infinity();

                auto get_value = [&](vertex v) {
                    if (!ebt::in(v, extra)) {
                        return -inf;
                    } else {
                        return extra.at(v).value;
                    }
                };

                for (auto& u: order) {
                    double max = get_value(u);
                    edge argmax;
                    bool update = false;

                    std::vector<edge> edges = f.in_edges(u);
                    std::vector<double> candidate_value;
                    candidate_value.resize(edges.size());

                    #pragma omp parallel for
                    for (int i = 0; i < edges.size(); ++i) {
                        edge& e = edges[i];
                        vertex v = f.tail(e);
                        candidate_value[i] = get_value(v) + f.weight(e);
                    }

                    for (int i = 0; i < edges.size(); ++i) {
                        if (candidate_value[i] > max) {
                            max = candidate_value[i];
                            argmax = edges[i];
                            update = true;
                        }
                    }

                    if (update) {
                        extra[u] = extra_data { argmax, max };
                    }
                }
            }

            std::vector<typename fst::edge> best_path(fst const& f)
            {
                double inf = std::numeric_limits<double>::infinity();
                double max = -inf;
                vertex argmax;

                for (auto v: f.finals()) {
                    if (ebt::in(v, extra) && extra.at(v).value > max) {
                        max = extra.at(v).value;
                        argmax = v;
                    }
                }

                std::vector<typename fst::edge> result;

                if (max == -inf) {
                    return result;
                }

                vertex u = argmax;

                std::vector<typename fst::vertex> const& initials = f.initials();
                std::unordered_set<vertex> initial_set { initials.begin(), initials.end() };

                while (!ebt::in(u, initial_set)) {
                    edge e = extra.at(u).pi;
                    vertex v = f.tail(e);
                    result.push_back(e);
                    u = v;
                }

                std::reverse(result.begin(), result.end());

                return result;
            }

        };

        template <class fst, class path_maker>
        fst shortest_path(fst const& f)
        {
            forward_one_best<fst> one_best;
            for (auto& v: f.initials()) {
                one_best.extra[v] = { edge_trait<typename fst::edge>::null, 0 };
            }
            one_best.merge(f, f.topo_order());
            std::vector<typename fst::edge> edges = one_best.best_path(f);
            return path_maker()(edges, f);
        }

    }

}

#endif
