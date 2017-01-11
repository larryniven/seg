#ifndef SEG_H
#define SEG_H

#include <vector>
#include "fst/fst.h"
#include "fst/ifst.h"
#include "nn/tensor-tree.h"

namespace seg {

    template <class fst>
    struct seg_weight {

        virtual ~seg_weight()
        {}

        virtual double operator()(fst const& f,
            typename fst::edge e) const = 0;

        virtual void accumulate_grad(double g, fst const& f,
            typename fst::edge e) const
        {}

        virtual void grad() const
        {}

    };

    template <class seg_data>
    struct seg_fst {

        using vertex = typename seg_data::vertex;
        using edge = typename seg_data::edge;
        using input_symbol = typename seg_data::input_symbol;
        using output_symbol = typename seg_data::output_symbol;

        seg_data const& data;

        std::vector<vertex> const& vertices() const;
        std::vector<edge> const& edges() const;
        vertex head(edge e) const;
        vertex tail(edge e) const;
        std::vector<edge> const& in_edges(vertex v) const;
        std::vector<edge> const& out_edges(vertex v) const;
        double weight(edge e) const;
        input_symbol const& input(edge e) const;
        output_symbol const& output(edge e) const;
        std::vector<vertex> const& initials() const;
        std::vector<vertex> const& finals() const;

        long time(vertex v) const;

    };

    struct iseg_data {
        using base_fst = ifst::fst;
        using vertex = int;
        using edge = int;
        using input_symbol = int;
        using output_symbol = int;

        std::shared_ptr<ifst::fst> fst;
        std::shared_ptr<std::vector<int>> topo_order;
        std::shared_ptr<seg_weight<ifst::fst>> weight_func;
        std::shared_ptr<seg_weight<ifst::fst>> cost_func;
        std::shared_ptr<tensor_tree::vertex> param;
    };

    struct pair_iseg_data {
        using base_fst = fst::pair_fst<ifst::fst, ifst::fst>;
        using vertex = std::tuple<int, int>;
        using edge = std::tuple<int, int>;
        using input_symbol = int;
        using output_symbol = int;

        std::shared_ptr<fst::pair_fst<ifst::fst, ifst::fst>> fst;
        std::shared_ptr<std::vector<std::tuple<int, int>>> topo_order;
        std::shared_ptr<seg_weight<fst::pair_fst<ifst::fst, ifst::fst>>> weight_func;
        std::shared_ptr<seg_weight<fst::pair_fst<ifst::fst, ifst::fst>>> cost_func;
        std::shared_ptr<tensor_tree::vertex> param;
    };

}

namespace fst {

    template <>
    struct edge_trait<int> {
        static int null;
    };

    template <>
    struct edge_trait<std::tuple<int, int>> {
        static std::tuple<int, int> null;
    };

}

#include "seg/seg-impl.h"

#endif
