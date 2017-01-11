#ifndef SEG_WEIGHT_H
#define SEG_WEIGHT_H

#include "seg/seg.h"
#include "fst/fst.h"
#include "fst/ifst.h"
#include "nn/tensor-tree.h"
#include <vector>
#include <memory>

namespace seg {

    template <class fst>
    struct composite_weight
        : public seg_weight<fst> {

        std::vector<std::shared_ptr<seg_weight<fst>>> weights;

        virtual double operator()(fst const& f,
            typename fst::edge e) const override;

        virtual void accumulate_grad(double g, fst const& f,
            typename fst::edge e) const override;

        virtual void grad() const override;

    };

    struct mode2_weight
        : public seg_weight<fst::pair_fst<ifst::fst, ifst::fst>> {

        std::shared_ptr<seg_weight<ifst::fst>> weight;

        mode2_weight(std::shared_ptr<seg_weight<ifst::fst>> weight);

        virtual double operator()(fst::pair_fst<ifst::fst, ifst::fst> const& fst,
            std::tuple<int, int> e) const override;

        virtual void accumulate_grad(double g, fst::pair_fst<ifst::fst, ifst::fst> const& fst,
            std::tuple<int, int> e) const override;

        virtual void grad() const override;
    };

    struct segrnn_score
        : public seg_weight<ifst::fst> {

        std::shared_ptr<tensor_tree::vertex> param;
        std::shared_ptr<autodiff::op_t> frames;
        std::shared_ptr<autodiff::op_t> pre_left;
        std::shared_ptr<autodiff::op_t> pre_right;
        std::shared_ptr<autodiff::op_t> left_end;
        std::shared_ptr<autodiff::op_t> right_end;
        std::shared_ptr<autodiff::op_t> pre_label;
        std::shared_ptr<autodiff::op_t> pre_length;

        std::shared_ptr<autodiff::op_t> score;

        std::vector<int> topo_order_shift;

        mutable std::default_random_engine *gen;
        double dropout;

        mutable std::vector<std::shared_ptr<autodiff::op_t>> edge_scores;

        segrnn_score(std::shared_ptr<tensor_tree::vertex> param,
            std::shared_ptr<autodiff::op_t> frames);

        segrnn_score(std::shared_ptr<tensor_tree::vertex> param,
            std::shared_ptr<autodiff::op_t> frames,
            double dropout,
            std::default_random_engine *gen);

        virtual double operator()(ifst::fst const& f,
            int e) const override;

        virtual void accumulate_grad(double g, ifst::fst const& f,
            int e) const override;

        virtual void grad() const override;

    };

}

#include "seg/seg-weight-impl.h"

#endif
