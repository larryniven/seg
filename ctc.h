#ifndef CTC_H
#define CTC_H

#include "fst/ifst.h"
#include "seg/seg.h"
#include "seg/loss.h"

namespace ctc {

    ifst::fst make_frame_fst(int nframes,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label);

    ifst::fst make_label_fst(std::vector<std::string> const& label_seq,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label);

    struct label_weight
        : public seg::seg_weight<ifst::fst> {

        std::vector<std::shared_ptr<autodiff::op_t>> const& label_score;

        label_weight(std::vector<std::shared_ptr<autodiff::op_t>> const& label_score);

        virtual double operator()(ifst::fst const& f, int e) const override;

        virtual void accumulate_grad(double g, ifst::fst const& f,
            int e) const override;
    };

    struct loss_func
        : public seg::loss_func {

        seg::iseg_data const& graph_data;
        ifst::fst label_graph;
        seg::pair_iseg_data pair_data;

        double logZ;

        fst::forward_log_sum<seg::seg_fst<seg::pair_iseg_data>> forward;
        fst::backward_log_sum<seg::seg_fst<seg::pair_iseg_data>> backward;

        loss_func(seg::iseg_data const& graph_data,
            std::vector<std::string> const& label_seq);

        virtual double loss() const override;
        virtual void grad(double scale=1) const override;
    };

}

#endif
