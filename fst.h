#ifndef FST_H
#define FST_H

#include "scrf/util.h"
#include <unordered_map>
#include <tuple>
#include <vector>
#include <limits>
#include <algorithm>
#include "ebt/ebt.h"
#include <memory>

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

    template <class fst_type>
    struct lazy_k_best {

        using vertex_type = typename fst_type::vertex_type;
        using edge_type = typename fst_type::edge_type;

        struct deck_entry {
            edge_type edge;
            int index;
            real value;
        };

        struct extra_data {
            std::unordered_map<edge_type, int> pi_deck_indices;
            std::vector<deck_entry> deck;
            bool no_more;
        };

        std::unordered_map<vertex_type, extra_data> extra;

        std::vector<vertex_type> ask(fst_type& fst, std::vector<vertex_type> stack)
        {
            std::unordered_set<vertex_type> result_set {stack.begin(), stack.end()};
            std::vector<vertex_type> result = stack;

            while (stack.size() != 0) {

                vertex_type u = stack.back();
                stack.pop_back();

                auto get_extra = [&](vertex_type const& v) -> extra_data& {
                    if (!ebt::in(v, extra)) {
                        extra[v].no_more = false;
                    }
                    return extra.at(v);
                };

                extra_data& u_data = get_extra(u);

                auto get_pi_ind = [&](edge_type const& e) {
                    return ebt::get(u_data.pi_deck_indices, e, -1);
                };

                for (auto&& e: fst.in_edges(u)) {
                    edge_type v = fst.tail(e);
                    extra_data& v_data = get_extra(v);

                    if (!ebt::in(v, result_set)
                            && get_pi_ind(e) == int(v_data.deck.size()) - 1
                            && !v_data.no_more) {
                        stack.push_back(v);
                        result_set.insert(v);
                        result.push_back(v);
                    }
                }
            }

            return result;
        }

        void merge(fst_type& fst, std::vector<vertex_type> stack)
        {
            while (stack.size() != 0) {
                vertex_type u = stack.back();
                stack.pop_back();

                auto get_extra = [&](vertex_type const& v) -> extra_data& {
                    if (!ebt::in(v, extra)) {
                        extra[v].no_more = false;
                    }
                    return extra.at(v);
                };

                extra_data& u_data = get_extra(u);

                auto get_pi_ind = [&](edge_type const& e) {
                    return ebt::get(u_data.pi_deck_indices, e, -1);
                };

                if (u_data.no_more) {
                    continue;
                }

                std::vector<edge_type> pi = fst.in_edges(u);

                if (pi.size() > 0) {
                    real max = -std::numeric_limits<real>::infinity();
                    edge_type argmax;

                    for (auto&& e: pi) {
                        edge_type v = fst.tail(e);
                        extra_data& v_data = get_extra(v);

                        if (get_pi_ind(e) + 1 < int(v_data.deck.size())
                                && v_data.deck.at(get_pi_ind(e) + 1).value + fst.weight(e) > max) {
                            max = v_data.deck.at(get_pi_ind(e) + 1).value + fst.weight(e);
                            argmax = e;
                        }
                    }

                    if (max == -std::numeric_limits<real>::infinity()) {
                        bool dead_end = false;
                        for (auto&& e: pi) {
                            edge_type v = fst.tail(e);
                            extra_data& v_data = get_extra(v);

                            if (v_data.deck.at(get_pi_ind(e) + 1).value == -std::numeric_limits<real>::infinity()) {
                                dead_end = true;
                            }
                        }

                        if (dead_end) {
                            continue;
                        }

                        for (auto&& e: pi) {
                            edge_type v = fst.tail(e);
                            extra_data& v_data = get_extra(v);

                            std::cout << fst.tail(e) << " " << fst.head(e)
                                << " " << fst.in_edges(fst.tail(e)).size()
                                << " " << fst.input(e) << " " << get_pi_ind(e) + 1
                                << " " << int(v_data.deck.size())
                                << " " << v_data.no_more;
                            if (get_pi_ind(e) + 1 < int(v_data.deck.size())) {
                                std::cout << " " << v_data.deck.at(get_pi_ind(e) + 1).value
                                    << " " << fst.weight(e);
                            }
                            std::cout << std::endl;
                        }
                        exit(1);
                    }

                    u_data.pi_deck_indices[argmax] += 1;
                    u_data.deck.push_back({argmax, u_data.pi_deck_indices.at(argmax), max});

                    vertex_type argmax_v = fst.tail(argmax);
                    extra_data& argmax_v_data = get_extra(argmax_v);
                    if (get_pi_ind(argmax) < int(argmax_v_data.deck.size()) - 1 || !argmax_v_data.no_more) {
                        bool has_more = false;

                        for (auto&& e: pi) {
                            vertex_type v = fst.tail(e);
                            extra_data& v_data = get_extra(v);

                            if (get_pi_ind(e) < int(v_data.deck.size()) - 1 || !v_data.no_more) {
                                has_more = true;
                                break;
                            }
                        }

                        u_data.no_more = !has_more;
                    }
                } else if (u_data.deck.size() == 0) {
                    deck_entry d_entry;
                    d_entry.index = -1;
                    d_entry.value = -std::numeric_limits<real>::infinity();
                    u_data.deck.push_back(d_entry);
                }

                if (pi.size() == 0) {
                    u_data.no_more = true;
                }
            }
        }

        void first_best(fst_type& fst, std::vector<vertex_type> finals, std::vector<vertex_type> initials)
        {
            std::vector<vertex_type>& stack = finals;
            stack = ask(fst, stack);

            for (auto& v: initials) {
                extra[v].deck.push_back({edge_type{}, -1, 0});
            }

            merge(fst, stack);
        }

        void next_best(fst_type& fst, std::vector<vertex_type> finals)
        {
            std::vector<vertex_type>& stack = finals;
            stack = ask(fst, stack);

            merge(fst, stack);
        }

        path<fst_type> best_path(fst_type& fst,
            std::vector<std::tuple<typename fst_type::vertex_type, int>> stack,
            std::vector<typename fst_type::vertex_type> initials)
        {
            using vertex_type = typename fst_type::vertex_type;
            using edge_type = typename fst_type::edge_type;

            path_data<fst_type> result;
            result.base_fst = &fst;

            while (stack.size() > 0) {
                vertex_type u = std::get<0>(stack.back());
                int deck_index = std::get<1>(stack.back());
                stack.pop_back();

                result.vertices.push_back(u);

                for (auto& s: initials) {
                    if (s == u) {
                        goto done;
                    }
                }

                edge_type e = extra.at(u).deck.at(deck_index).edge;

                vertex_type v = fst.tail(e);
                real value = extra.at(u).deck.at(deck_index).value;

                result.edges.push_back(e);
                result.in_edges[u].push_back(e);
                result.out_edges[v].push_back(e);

                for (int i = 0; i < extra.at(v).deck.size(); ++i) {
                    if (std::abs(extra.at(v).deck.at(i).value - (value - fst.weight(e)))
                            <= std::min(std::abs(extra.at(v).deck.at(i).value),
                                std::abs(value - fst.weight(e))) * 1e-3) {
                        stack.push_back(std::make_tuple(v, i));
                        break;
                    }
                }
            }

            done:

            // TODO: dangerous
            result.initial = fst.initial();
            result.final = fst.final();

            std::reverse(result.edges.begin(), result.edges.end());

            path<fst_type> p;
            p.data = std::make_shared<path_data<fst_type>>(std::move(result));

            return p;
        }

        path<fst_type> best_path(fst_type& fst)
        {
            return best_path(fst, {std::make_tuple(fst.final(), 0)}, {fst.initial()});
        }

        sub_fst<fst_type> backtrack(fst_type& fst,
            std::vector<std::tuple<typename fst_type::vertex_type, int>> stack,
            std::vector<typename fst_type::vertex_type> initials)
        {
            using vertex_type = typename fst_type::vertex_type;
            using edge_type= typename fst_type::edge_type;

            sub_fst_data<fst_type> result;
            result.base_fst = &fst;

            while (stack.size() > 0) {
                vertex_type u = std::get<0>(stack.back());
                int deck_index = std::get<1>(stack.back());
                stack.pop_back();

                result.vertices.push_back(u);

                for (auto& s: initials) {
                    if (s == u) {
                        continue;
                    }
                }

                edge_type e = extra.at(u).deck.at(deck_index).edge;

                vertex_type v = fst.tail(e);
                real value = extra.at(u).deck.at(deck_index).value;

                result.edges.push_back(e);
                result.in_edges[u].push_back(e);
                result.out_edges[v].push_back(e);

                for (int i = 0; i < extra.at(v).deck.size(); ++i) {
                    if (std::abs(extra.at(v).deck.at(i).value - (value - fst.weight(e)))
                            <= std::min(std::abs(extra.at(v).deck.at(i).value), std::abs(value - fst.weight(e))) * 1e-3) {
                        stack.push_back(std::make_tuple(v, i));
                    }
                }
            }

            // TODO: dangerous
            result.initial = fst.data.initial;
            result.final = fst.data.final;

            std::reverse(result.edges.begin(), result.edges.end());

            sub_fst<fst_type> f;
            f.data = std::make_shared<sub_fst_data<fst_type>>(std::move(result));

            return f;
        }

        sub_fst<fst_type> backtrack(fst_type& fst,
            std::vector<std::tuple<typename fst_type::vertex_type, int>> stack)
        {
            return backtrack(fst, stack, {fst.initial()});
        }

        sub_fst_data<fst_type> backtrack(fst_type& fst)
        {
            std::vector<std::tuple<typename fst_type::vertex_type, int>> stack;
            stack.push_back(std::make_tuple(fst.final(), 0));
            return backtrack(fst, stack);
        }

    };

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

        // mutable std::unordered_map<vertex_type, std::vector<edge_type>> pi_cache_;

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
            // if (ebt::in(v, pi_cache_)) {
            //     return pi_cache_.at(v);
            // }

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

            // pi_cache_[v] = std::move(result);
            // return pi_cache_.at(v);

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
            return fst3->output(std::get<1>(e));
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

            to_expand.insert(fst.initial(), 0);
            score[fst.initial()] = 0;

            real inf = std::numeric_limits<real>::infinity();

            while (to_expand.size()) {
                ebt::MaxHeap<vertex_type, real> expanded;

                while (to_expand.size()) {
                    auto v = to_expand.extract_max();

                    for (auto& e: fst.out_edges(v)) {
                        vertex_type const& head = fst.head(e);
                        real s = fst.weight(e) + score.at(v);
                        if (s > ebt::get(score, head, -inf)) {
                            expanded.insert(head, s);
                            score[head] = s;
                            pi[head] = e;
                        }
                    }
                }

                for (int i = 0; i < top_k && expanded.size() > 0; ++i) {
                    auto v = expanded.extract_max();
                    to_expand.insert(v, score.at(v));
                }
            }
        }

        path<fst_type> backtrack(fst_type& fst, vertex_type const& v)
        {
            path_data<fst_type> result { fst };

            result.final = v;
            result.vertices.push_back(v);

            auto add_edge = [&](edge_type const& e) {
                result.vertices.push_back(fst.tail(e));
                result.edges.push_back(e);
                result.in_edges[fst.head(e)].push_back(e);
                result.out_edges[fst.tail(e)].push_back(e);
            };

            auto u = v;
            while (ebt::in(u, pi)) {
                add_edge(pi.at(u));
                u = fst.tail(pi.at(u));
            }

            result.initial = u;

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

}

#endif
