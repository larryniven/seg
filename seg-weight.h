#ifndef SEG_WEIGHT_H
#define SEG_WEIGHT_H

#include "seg/seg.h"
#include "fst/fst.h"
#include "fst/ifst.h"
#include "nn/tensor-tree.h"
#include <vector>
#include <memory>

namespace seg {

    template <class fst_t, class T>
    struct weight_wrapper
        : public seg_weight<fst_t> {

        T func;

        weight_wrapper(T func);
        
        virtual double operator()(fst_t const& f,
            typename fst_t::edge e) const override;
    };

    template <class fst_t, class T>
    std::shared_ptr<weight_wrapper<fst_t, T>> make_weight(T&& t);

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

    template <class fst>
    struct cached_weight
        : public seg_weight<fst> {

        std::shared_ptr<seg_weight<fst>> weight;

        mutable std::unordered_map<typename fst::edge, double> cache;

        cached_weight(std::shared_ptr<seg_weight<fst>> weight);

        virtual double operator()(fst const& f,
            typename fst::edge e) const override;

        virtual void accumulate_grad(double g, fst const& f,
            typename fst::edge e) const override;

        virtual void grad() const override;

    };

    struct mode1_weight
        : public seg_weight<fst::pair_fst<ifst::fst, ifst::fst>> {

        std::shared_ptr<seg_weight<ifst::fst>> weight;

        mode1_weight(std::shared_ptr<seg_weight<ifst::fst>> weight);

        virtual double operator()(fst::pair_fst<ifst::fst, ifst::fst> const& fst,
            std::tuple<int, int> e) const override;

        virtual void accumulate_grad(double g, fst::pair_fst<ifst::fst, ifst::fst> const& fst,
            std::tuple<int, int> e) const override;

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

    struct frame_sum_score
        : public seg_weight<ifst::fst> {

        std::shared_ptr<autodiff::op_t> frames;

        frame_sum_score(std::shared_ptr<autodiff::op_t> frames);

        virtual double operator()(ifst::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ifst::fst const& f,
            int e) const override;

    };

    struct frame_avg_score
        : public seg_weight<ifst::fst> {

        std::shared_ptr<autodiff::op_t> param;
        std::shared_ptr<autodiff::op_t> frames;
        std::shared_ptr<autodiff::op_t> score;

        frame_avg_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames);

        virtual double operator()(ifst::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ifst::fst const& f,
            int e) const override;

        virtual void grad() const override;

    };

    struct frame_samples_score
        : public seg_weight<ifst::fst> {

        std::shared_ptr<autodiff::op_t> param;
        std::shared_ptr<autodiff::op_t> frames;
        std::shared_ptr<autodiff::op_t> score;
        double scale;

        frame_samples_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames, double scale);

        virtual double operator()(ifst::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ifst::fst const& f,
            int e) const override;

        virtual void grad() const override;

    };

    struct left_boundary_score
        : public seg_weight<ifst::fst> {

        std::shared_ptr<autodiff::op_t> param;
        std::shared_ptr<autodiff::op_t> frames;
        std::shared_ptr<autodiff::op_t> score;
        int shift;

        left_boundary_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames, int shift);

        virtual double operator()(ifst::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ifst::fst const& f,
            int e) const override;

        virtual void grad() const override;

    };

    struct right_boundary_score
        : public seg_weight<ifst::fst> {

        std::shared_ptr<autodiff::op_t> param;
        std::shared_ptr<autodiff::op_t> frames;
        std::shared_ptr<autodiff::op_t> score;
        int shift;

        right_boundary_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames, int shift);

        virtual double operator()(ifst::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ifst::fst const& f,
            int e) const override;

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

        mutable int topo_shift;

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

    struct length_score
        : public seg_weight<ifst::fst> {

        std::shared_ptr<autodiff::op_t> param;

        length_score(std::shared_ptr<autodiff::op_t> param);

        virtual double operator()(ifst::fst const& f,
            int e) const override;

        virtual void accumulate_grad(double g, ifst::fst const& f,
            int e) const override;

    };

    struct bias0_score
        : public seg_weight<ifst::fst> {

        std::shared_ptr<autodiff::op_t> param;

        bias0_score(std::shared_ptr<autodiff::op_t> param);

        virtual double operator()(ifst::fst const& f,
            int e) const override;

        virtual void accumulate_grad(double g, ifst::fst const& f,
            int e) const override;

    };

    struct bias1_score
        : public seg_weight<ifst::fst> {

        std::shared_ptr<autodiff::op_t> param;

        bias1_score(std::shared_ptr<autodiff::op_t> param);

        virtual double operator()(ifst::fst const& f,
            int e) const override;

        virtual void accumulate_grad(double g, ifst::fst const& f,
            int e) const override;

    };

    struct logsoftmax_score
        : public seg_weight<ifst::fst> {

        std::shared_ptr<autodiff::op_t> param;
        std::shared_ptr<autodiff::op_t> frames;
        std::shared_ptr<autodiff::op_t> first;
        std::shared_ptr<autodiff::op_t> score;
        std::shared_ptr<autodiff::op_t> prob;

        logsoftmax_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames);

        virtual double operator()(ifst::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ifst::fst const& f,
            int e) const override;

        virtual void grad() const override;

    };

    struct label_logsoftmax_score
        : public seg_weight<ifst::fst> {

        std::shared_ptr<autodiff::op_t> param;
        std::shared_ptr<autodiff::op_t> frames;
        std::shared_ptr<autodiff::op_t> score;
        std::shared_ptr<autodiff::op_t> prob;

        label_logsoftmax_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames);

        virtual double operator()(ifst::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ifst::fst const& f,
            int e) const override;

        virtual void grad() const override;

    };

    struct length_logsoftmax_score
        : public seg_weight<ifst::fst> {

        std::shared_ptr<autodiff::op_t> param;
        std::shared_ptr<autodiff::op_t> frames;
        std::shared_ptr<autodiff::op_t> score;
        std::shared_ptr<autodiff::op_t> prob;

        length_logsoftmax_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames);

        virtual double operator()(ifst::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ifst::fst const& f,
            int e) const override;

        virtual void grad() const override;

    };

    struct label_tanh_score
        : public seg_weight<ifst::fst> {

        std::shared_ptr<autodiff::op_t> param;
        std::shared_ptr<autodiff::op_t> frames;
        std::shared_ptr<autodiff::op_t> score;
        std::shared_ptr<autodiff::op_t> prob;

        label_tanh_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames);

        virtual double operator()(ifst::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ifst::fst const& f,
            int e) const override;

        virtual void grad() const override;

    };

    struct length_tanh_score
        : public seg_weight<ifst::fst> {

        std::shared_ptr<autodiff::op_t> param;
        std::shared_ptr<autodiff::op_t> frames;
        std::shared_ptr<autodiff::op_t> score;
        std::shared_ptr<autodiff::op_t> prob;

        length_tanh_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames);

        virtual double operator()(ifst::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ifst::fst const& f,
            int e) const override;

        virtual void grad() const override;

    };

}

#include "seg/seg-weight-impl.h"

#endif
