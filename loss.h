#ifndef LOSS_H
#define LOSS_H

#include "seg/seg.h"
#include "fst/fst-algo.h"
#include "seg/loss-util.h"

namespace seg {

    struct loss_func {
        virtual ~loss_func();

        virtual double loss() const = 0;
        virtual void grad() const = 0;
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
            std::vector<int> const& label_seq);

        virtual double loss() const override;

        virtual void grad() const override;

    };

    struct entropy_loss
        : public loss_func {

        iseg_data& graph_data;
        double logZ;
        double exp_score;

        fst::forward_log_sum<seg_fst<iseg_data>> forward_graph;
        fst::backward_log_sum<seg_fst<iseg_data>> backward_graph;

        forward_exp_score<seg_fst<iseg_data>> forward_exp;
        backward_exp_score<seg_fst<iseg_data>> backward_exp;

        entropy_loss(iseg_data& graph_data);

        virtual double loss() const override;

        virtual void grad() const override;

    };

}


#endif
