#include "scrf/scrf_feat.h"
#include <cassert>

namespace scrf {

    composite_feature::composite_feature(std::string name)
        : name_(name)
    {}

    int composite_feature::size() const
    {
        int sum = 0;
        for (auto& f: features) {
            sum += f->size();
        }
        return sum;
    }

    std::string composite_feature::name() const
    {
        return name_;
    }

    void composite_feature::operator()(
        param_t& feat,
        fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        for (auto& f: features) {
            (*f)(feat, fst, e);
        }
    }
        
    scrf_weight::~scrf_weight()
    {}

    lexicalized_feature::lexicalized_feature(
        int order,
        std::shared_ptr<segfeat::feature> feat_func,
        std::vector<std::vector<real>> const& frames)
        : order(order), feat_func(feat_func), frames(frames)
    {}

    int lexicalized_feature::size() const
    {
        return 0;
    }

    std::string lexicalized_feature::name() const
    {
        return "";
    }

    void lexicalized_feature::operator()(
        param_t& feat,
        fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        segfeat::feat_t raw_feat;

        lattice::fst const& lat = *fst.fst1;
        int tail_time = lat.data->vertices.at(std::get<0>(fst.tail(e))).time;
        int head_time = lat.data->vertices.at(std::get<0>(fst.head(e))).time;

        (*feat_func)(raw_feat, frames, tail_time, head_time);

        std::string label_tuple;

        if (order == 0) {
            // do nothing
        } else if (order == 1) {
            label_tuple = fst.output(e);
        } else if (order == 2) {
            auto const& lm = *fst.fst2;
            label_tuple = lm.data->history.at(std::get<1>(fst.tail(e))) + "_" + fst.output(e);
        } else {
            std::cerr << "order " << order << " not implemented" << std::endl;
            exit(1);
        }

        feat.class_param[label_tuple] = std::move(raw_feat);
    }

    namespace feature {

        int lm_score::size() const
        {
            return 1;
        }

        std::string lm_score::name() const
        {
            return "lm-score";
        }

        void lm_score::operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            feat.class_param["[lm] shared"].push_back(fst.fst2->weight(std::get<1>(e)));
        }

        int lattice_score::size() const
        {
            return 1;
        }

        std::string lattice_score::name() const
        {
            return "lattice-score";
        }

        void lattice_score::operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            feat.class_param["[lattice] shared"].push_back(fst.fst1->weight(std::get<0>(e)));
        }

    }

}
