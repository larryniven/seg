#ifndef LOSS_H
#define LOSS_H

#include "seg/seg.h"
#include "fst/fst-algo.h"
#include "seg/loss-util.h"
#include "seg/seg-cost.h"

namespace seg {

    struct loss_func {
        virtual ~loss_func();

        virtual double loss() const = 0;
        virtual void grad(double scale=1) const = 0;
    };

    struct log_loss
        : public loss_func {

        iseg_data& graph_data;
        std::vector<int> min_cost_path;
        std::vector<cost::segment<int>> min_cost_segs;

        fst::forward_log_sum<seg_fst<iseg_data>> forward;
        fst::backward_log_sum<seg_fst<iseg_data>> backward;

        log_loss(iseg_data& graph_data,
            std::vector<cost::segment<int>> const& gt_segs,
            std::vector<int> const& sils);

        virtual double loss() const override;

        virtual void grad(double scale=1) const override;

    };

    struct marginal_log_loss
        : public loss_func {

        iseg_data& graph_data;

        fst::forward_log_sum<seg_fst<iseg_data>> forward_graph;
        fst::backward_log_sum<seg_fst<iseg_data>> backward_graph;

        pair_iseg_data pair_data;

        fst::forward_log_sum<seg_fst<pair_iseg_data>> forward_label;
        fst::backward_log_sum<seg_fst<pair_iseg_data>> backward_label;

        marginal_log_loss(iseg_data& graph_data,
            ifst::fst& label_fst);

        virtual double loss() const override;

        virtual void grad(double scale=1) const override;

    };

    struct weight_risk
        : public risk_func<seg_fst<iseg_data>> {

        virtual double operator()(seg_fst<iseg_data> const& fst, int e) const;
    };

    struct entropy_loss
        : public loss_func {

        iseg_data& graph_data;
        double logZ;
        double exp_score;

        fst::forward_log_sum<seg_fst<iseg_data>> forward_graph;
        fst::backward_log_sum<seg_fst<iseg_data>> backward_graph;

        forward_exp_risk<seg_fst<iseg_data>> forward_exp;
        backward_exp_risk<seg_fst<iseg_data>> backward_exp;

        entropy_loss(iseg_data& graph_data);

        virtual double loss() const override;

        virtual void grad(double scale=1) const override;

    };

    struct empirical_bayes_risk
        : public loss_func {

        iseg_data& graph_data;
        std::shared_ptr<risk_func<seg_fst<iseg_data>>> risk;

        double logZ;
        double exp_risk;

        fst::forward_log_sum<seg_fst<iseg_data>> f_log_sum;
        fst::backward_log_sum<seg_fst<iseg_data>> b_log_sum;

        forward_exp_risk<seg_fst<iseg_data>> f_risk;
        backward_exp_risk<seg_fst<iseg_data>> b_risk;

        empirical_bayes_risk(iseg_data& graph_data,
            std::shared_ptr<risk_func<seg_fst<iseg_data>>> risk);

        virtual double loss() const override;

        virtual void grad(double scale=1) const override;

    };

#if 0
    struct frame_reconstruction_risk
        : public risk_func<seg_fst<iseg_data>> {

        std::vector<std::shared_ptr<autodiff::op_t>> const& frames;
        std::shared_ptr<tensor_tree::vertex> param;

        mutable std::vector<std::shared_ptr<autodiff::op_t>> risk_cache;

        std::vector<int> topo_order_shift;

        frame_reconstruction_risk(
            std::vector<std::shared_ptr<autodiff::op_t>> const& frames,
            std::shared_ptr<tensor_tree::vertex> param);

        virtual double operator()(seg_fst<iseg_data> const& fst, int e) const;

        virtual void accumulate_grad(double g, seg_fst<iseg_data> const& fst, int e);

        virtual void grad();
    };
#endif

}


#endif
