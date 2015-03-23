#ifndef FST_H
#define FST_H

#include <unordered_map>
#include <tuple>
#include <vector>
#include <limits>
#include <algorithm>
#include "ebt/ebt.h"

namespace fst {

    template <class fst_type>
    struct lazy_k_best {

        using vertex_type = typename fst_type::vertex_type;
        using edge_type = typename fst_type::edge_type;

        struct deck_entry {
            edge_type edge;
            int index;
            double value;
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
                    double max = -std::numeric_limits<double>::infinity();
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

                    if (max == -std::numeric_limits<double>::infinity()) {
                        bool dead_end = false;
                        for (auto&& e: pi) {
                            edge_type v = fst.tail(e);
                            extra_data& v_data = get_extra(v);

                            if (v_data.deck.at(get_pi_ind(e) + 1).value == -std::numeric_limits<double>::infinity()) {
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
                    d_entry.value = -std::numeric_limits<double>::infinity();
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

        double deck_at(vertex_type const& v, int k)
        {
            return extra.at(v).deck.at(k).value;
        }
    };

    template <class fst>
    struct sub_fst_data {
        using vertex_type = typename fst::vertex_type;
        using edge_type = typename fst::edge_type;

        fst& base_fst;
        std::vector<vertex_type> vertices;
        std::vector<edge_type> edges;
        std::unordered_map<vertex_type, std::vector<edge_type>> in_edges;
        std::unordered_map<vertex_type, std::vector<edge_type>> out_edges;
        vertex_type initial;
        vertex_type final;
    };

    template <class fst>
    struct sub_fst {
        using vertex_type = typename fst::vertex_type;
        using edge_type = typename fst::edge_type;

        sub_fst_data<fst>& data;

        std::vector<vertex_type> vertices() const
        {
            return data.vertices;
        }

        std::vector<edge_type> edges() const
        {
            return data.edges;
        }

        double weight(edge_type const& e) const
        {
            return data.base_fst.weight(e);
        }

        vertex_type tail(edge_type const& e) const
        {
            return data.base_fst.tail(e);
        }

        vertex_type head(edge_type const& e) const
        {
            return data.base_fst.head(e);
        }

        std::vector<edge_type> in_edges(vertex_type const& v) const
        {
            return data.in_edges.at(v);
        }

        std::vector<edge_type> out_edges(vertex_type const& v) const
        {
            return data.out_edges.at(v);
        }

        vertex_type initial() const
        {
            return data.initial;
        }

        vertex_type final() const
        {
            return data.final;
        }

        std::string input(edge_type const& e) const
        {
            return data.base_fst.input(e);
        }

        std::string output(edge_type const& e) const
        {
            return data.base_fst.output(e);
        }

    };

    template <class fst> using path_data = sub_fst_data<fst>;
    template <class fst> using path = sub_fst<fst>;

    template <class lazy_k_best, class fst_type>
    path_data<fst_type> best_path(lazy_k_best const& k_best, fst_type& fst,
        std::vector<std::tuple<typename fst_type::vertex_type, int>> stack,
        std::vector<typename fst_type::vertex_type> initials)
    {
        using vertex_type = typename fst_type::vertex_type;
        using edge_type = typename fst_type::edge_type;

        path_data<fst_type> result { fst };

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

            edge_type e = k_best.extra.at(u).deck.at(deck_index).edge;

            vertex_type v = fst.tail(e);
            double value = k_best.extra.at(u).deck.at(deck_index).value;

            result.edges.push_back(e);
            result.in_edges[u].push_back(e);
            result.out_edges[v].push_back(e);

            for (int i = 0; i < k_best.extra.at(v).deck.size(); ++i) {
                if (std::abs(k_best.extra.at(v).deck.at(i).value - (value - fst.weight(e)))
                        <= std::min(std::abs(k_best.extra.at(v).deck.at(i).value), std::abs(value - fst.weight(e))) * 1e-3) {
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

        return result;
    }

    template <class lazy_k_best, class fst_type>
    path_data<fst_type> best_path(lazy_k_best const& k_best, fst_type& fst)
    {
        return best_path(k_best, fst, {std::make_tuple(fst.final(), 0)}, {fst.initial()});
    }

    template <class fst_type>
    sub_fst_data<fst_type> backtrack(lazy_k_best<fst_type> const& k_best, fst_type& fst,
        std::vector<std::tuple<typename fst_type::vertex_type, int>> stack,
        std::vector<typename fst_type::vertex_type> initials)
    {
        using vertex_type = typename fst_type::vertex_type;
        using edge_type= typename fst_type::edge_type;

        sub_fst_data<fst_type> result;

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

            edge_type e = k_best.extra.at(u).deck.at(deck_index).edge;

            vertex_type v = fst.tail(e);
            double value = k_best.extra.at(u).deck.at(deck_index).value;

            result.edges.push_back(e);
            result.in_edges[u].push_back(e);
            result.out_edges[v].push_back(e);

            for (int i = 0; i < k_best.extra.at(v).deck.size(); ++i) {
                if (std::abs(k_best.extra.at(v).deck.at(i).value - (value - fst.weight(e)))
                        <= std::min(std::abs(k_best.extra.at(v).deck.at(i).value), std::abs(value - fst.weight(e))) * 1e-3) {
                    stack.push_back(std::make_tuple(v, i));
                }
            }
        }

        // TODO: dangerous
        result.initial = fst.data.initial;
        result.final = fst.data.final;

        std::reverse(result.edges.begin(), result.edges.end());

        return result;
    }

    template <class fst_type>
    sub_fst_data<fst_type> backtrack(lazy_k_best<fst_type> const& k_best, fst_type& fst,
        std::vector<std::tuple<typename fst_type::vertex_type, int>> stack)
    {
        return backtrack(k_best, fst, stack, {fst.initial()});
    }

    template <class fst_type>
    sub_fst_data<fst_type> backtrack(fst_type& fst)
    {
        std::vector<std::tuple<typename fst_type::vertex_type, int>> stack;
        stack.push_back(std::make_tuple(fst.final(), 0));
        return backtrack(fst, stack);
    }

    template <class fst_type1, class fst_type2>
    struct composed_fst {
        using edge_type = std::tuple<typename fst_type1::edge_type,
            typename fst_type2::edge_type>;
        using vertex_type = std::tuple<typename fst_type1::vertex_type,
            typename fst_type2::vertex_type>;

        fst_type1& fst1;
        fst_type2& fst2;
        mutable std::unordered_map<vertex_type, std::vector<edge_type>> pi_cache_;

        std::vector<vertex_type> vertices() const
        {
            std::vector<vertex_type> result;

            for (auto& v1: fst1.vertices()) {
                for (auto& v2: fst2.vertices()) {
                    result.push_back(std::make_tuple(v1, v2));
                }
            }

            return result;
        }

        std::vector<edge_type> edges() const
        {
            std::vector<edge_type> result;

            for (auto& e1: fst1.edges()) {
                for (auto& e2: fst2.edges()) {
                    if (fst1.output(e1) == fst2.input(e2)) {
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

            auto& e2_set = fst2.in_edges_map(std::get<1>(v));

            for (auto& e1: fst1.in_edges(std::get<0>(v))) {
                if (!ebt::in(fst1.output(e1), e2_set)) {
                    continue;
                }

                for (auto& e2: e2_set.at(fst1.output(e1))) {
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

            auto& e2_set = fst2.out_edges_map(std::get<1>(v));

            for (auto& e1: fst1.out_edges(std::get<0>(v))) {
                if (!ebt::in(fst1.output(e1), e2_set)) {
                    continue;
                }

                for (auto& e2: e2_set.at(fst1.output(e1))) {
                    result.push_back(std::make_tuple(e1, e2));
                }
            }

            return result;
        }

        double weight(edge_type const& e) const
        {
            return fst1.weight(std::get<0>(e)) + fst2.weight(std::get<1>(e));
        }

        vertex_type tail(edge_type const& e) const
        {
            return std::make_tuple(fst1.tail(std::get<0>(e)),
                fst2.tail(std::get<1>(e)));
        }

        vertex_type head(edge_type const& e) const
        {
            return std::make_tuple(fst1.head(std::get<0>(e)),
                fst2.head(std::get<1>(e)));
        }

        vertex_type initial() const
        {
            return std::make_tuple(fst1.initial(), fst2.initial());
        }

        vertex_type final() const
        {
            return std::make_tuple(fst1.final(), fst2.final());
        }

        std::string input(edge_type const& e) const
        {
            return fst1.input(std::get<0>(e));
        }

        std::string output(edge_type const& e) const
        {
            return fst2.output(std::get<1>(e));
        }

    };

    template <class fst_type>
    struct shortest_path_lagrange_relaxation {
        using vertex_type = typename fst_type::vertex_type;
        using edge_type = typename fst_type::edge_type;

        fst_type& fst;

        std::unordered_map<edge_type, double> weight_cache;
        std::unordered_map<vertex_type, double> dual_var;
        std::unordered_map<edge_type, double> primal_var;

        double weight(edge_type const& e)
        {
            if (!ebt::in(e, weight_cache)) {
                weight_cache[e] = fst.weight(e);
            }
            return weight_cache.at(e);
        }

        void solve_lagrangian()
        {
            for (auto& p: primal_var) {
                auto& e = p.first;
                primal_var.at(e) = (weight(e) + dual_var.at(fst.head(e)) - dual_var.at(fst.tail(e))) > 0 ? 1 : 0;
            }
        }

        void update_multiplier()
        {
            double eta = 1e-2;

            for (auto& p: primal_var) {
                dual_var.at(fst.tail(p.first)) += eta * p.second;
                dual_var.at(fst.head(p.first)) -= eta * p.second;
            }

            dual_var.at(fst.initial()) -= eta;
            dual_var.at(fst.final()) += eta;
        }

        double dual_obj()
        {
            double sum = 0;
            for (auto& p: primal_var) {
                auto& e = p.first;
                sum += weight(e) + dual_var.at(fst.head(e)) - dual_var.at(fst.tail(e));
            }
            sum += dual_var.at(fst.initial());
            sum -= dual_var.at(fst.final());

            return sum;
        }

        void solve()
        {
            int max_iter = 10;

            for (auto& e: fst.edges()) {
                primal_var[e] = 0;
                dual_var[fst.tail(e)] = 0;
                dual_var[fst.head(e)] = 0;
            }

            for (int iter = 0; iter < max_iter; ++iter) {
                std::cout << "solve lagrangian" << std::endl;
                solve_lagrangian();
                std::cout << "update multiplier" << std::endl;
                update_multiplier();
                std::cout << "dual obj: " << dual_obj() << std::endl;
            }
        }

    };

    template <class fst_type>
    struct beam_search {
        using vertex_type = typename fst_type::vertex_type;
        using edge_type = typename fst_type::edge_type;

        std::unordered_map<vertex_type, double> score;
        std::unordered_map<vertex_type, edge_type> pi;

        void search(fst_type const& fst, int top_k)
        {
            ebt::MaxHeap<vertex_type, double> to_expand;

            to_expand.insert(fst.initial(), 0);
            score[fst.initial()] = 0;

            double inf = std::numeric_limits<double>::infinity();

            while (to_expand.size()) {
                ebt::MaxHeap<vertex_type, double> expanded;

                while (to_expand.size()) {
                    auto v = to_expand.extract_max();

                    for (auto& e: fst.out_edges(v)) {
                        vertex_type const& head = fst.head(e);
                        double s = fst.weight(e) + score.at(v);
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

        path_data<fst_type> backtrack(fst_type& fst, vertex_type const& v)
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

            return result;
        }
    };

}

#endif
