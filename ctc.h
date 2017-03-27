#ifndef CTC_H
#define CTC_H

#include "fst/ifst.h"
#include "seg/seg.h"
#include "seg/loss.h"

namespace ctc {

    ifst::fst make_frame_fst(int nframes,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label);

    ifst::fst make_label_fst(std::vector<int> const& label_seq,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label);

    ifst::fst make_label_fst_hmm1s(std::vector<int> const& label_seq,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label);

    ifst::fst make_label_fst_hmm2s(std::vector<int> const& label_seq,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label);

    ifst::fst make_phone_fst(std::unordered_map<std::string, int> const& label_id,
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
            ifst::fst const& label_fst);

        virtual double loss() const override;
        virtual void grad(double scale=1) const override;
    };

    template <class fst_t>
    struct beam_search {

        struct extra_data {
            typename fst_t::vertex vertex;
            bool ends_blank;
            int seq_id;
        };

        std::vector<std::vector<int>> id_seq;
        std::unordered_map<std::vector<int>, int> seq_id;

        std::unordered_map<std::pair<bool, int>, double> path_score;
        std::vector<extra_data> heap;

        void search(fst_t const& f, typename fst_t::output_symbol blk, int topk);

    };

}

#include "ctc-impl.h"

#endif
