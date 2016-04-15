#ifndef FST_H
#define FST_H

#include <unordered_map>
#include <tuple>
#include <vector>
#include <limits>
#include <algorithm>
#include <memory>
#include "ebt/ebt.h"

namespace fst {

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

    /*
     * The class `symbol_trait` is useful for defining `eps`
     * and other special symbols.
     *
     */
    template <class symbol>
    struct symbol_trait;

    template <>
    struct symbol_trait<std::string> {
        static std::string eps;
    };

    /*
     * The class `edge_trait` is usefule for creating null edges.
     * Null edges are used, for example, in tracking back
     * the shortest path.
     *
     */
    template <class edge>
    struct edge_trait;

    /*
     * The return type is a `shared_ptr` so that subclasses of `fst`
     * can be returned and the data is managed.
     *
     */
    template <class fst>
    struct path_maker {
        virtual std::shared_ptr<fst> operator()(std::vector<typename fst::edge> const& edges,
            fst const& f) const = 0;
    };

    template <class fst>
    std::vector<typename fst::vertex> topo_order(fst const& f);

    template <class fst>
    struct forward_one_best {

        using vertex = typename fst::vertex;
        using edge = typename fst::edge;

        struct extra_data {
            edge pi;
            double value;
        };

        std::unordered_map<vertex, extra_data> extra;

        void merge(fst const& f, std::vector<vertex> const& order);

        std::vector<typename fst::edge> best_path(fst const& f);

    };

    template <class fst, class path_maker>
    std::shared_ptr<fst> shortest_path(fst const& f);

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

    template <class fst>
    void forward_one_best<fst>::merge(fst const& f, std::vector<typename fst::vertex> const& order)
    {
        double inf = std::numeric_limits<double>::infinity();

        auto get_value = [&](vertex v) {
            if (!ebt::in(v, extra)) {
                return -inf;
            } else {
                return extra.at(v).value;
            }
        };

        for (auto& u: order) {
            double max = get_value(u);
            typename fst::edge argmax;
            bool update = false;

            std::vector<edge> edges = f.in_edges(u);
            std::vector<double> candidate_value;
            candidate_value.resize(edges.size());

            #pragma omp parallel for
            for (int i = 0; i < edges.size(); ++i) {
                typename fst::edge& e = edges[i];
                typename fst::vertex v = f.tail(e);
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

    template <class fst>
    std::vector<typename fst::edge> forward_one_best<fst>::best_path(fst const& f)
    {
        double inf = std::numeric_limits<double>::infinity();
        double max = -inf;
        typename fst::vertex argmax;

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
        std::unordered_set<typename fst::vertex> initial_set { initials.begin(), initials.end() };

        while (!ebt::in(u, initial_set)) {
            edge e = extra.at(u).pi;
            vertex v = f.tail(e);
            result.push_back(e);
            u = v;
        }

        std::reverse(result.begin(), result.end());

        return result;
    }

    template <class fst, class path_maker>
    std::shared_ptr<fst> shortest_path(fst const& f)
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

#endif
