#include "scrf/scrf_feat.h"
#include <cassert>

namespace scrf {

    composite_feature::composite_feature(std::string name)
        : name_(name)
    {}

    void composite_feature::operator()(
        feat_t& feat,
        fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        for (auto& f: features) {
            (*f)(feat, fst, e);
        }
    }
        
    lexicalized_feature::lexicalized_feature(
        int order,
        std::shared_ptr<segfeat::feature> feat_func,
        std::vector<std::vector<real>> const& frames)
        : order(order), feat_func(feat_func), frames(frames)
    {}

    void lexicalized_feature::operator()(
        feat_t& feat,
        fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        segfeat::feat_t raw_feat;

        lattice::fst const& lat = *fst.fst1;
        int tail_time = lat.data->vertices.at(std::get<0>(fst.tail(e))).time;
        int head_time = lat.data->vertices.at(std::get<0>(fst.head(e))).time;

        (*feat_func)(raw_feat, frames, tail_time, head_time);

        std::string label_tuple = "[seg] ";

        if (order == 0) {
            label_tuple += "shared";
            // do nothing
        } else if (order == 1) {
            label_tuple += fst.output(e);
        } else if (order == 2) {
            auto const& lm = *fst.fst2;
            label_tuple += lm.data->history.at(std::get<1>(fst.tail(e))) + "_" + fst.output(e);
        } else {
            std::cerr << "order " << order << " not implemented" << std::endl;
            exit(1);
        }

        feat.class_param[label_tuple] = std::move(raw_feat);
    }

    namespace feature {

        void lm_score::operator()(
            feat_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            feat.class_param["[lm] shared"].push_back(fst.fst2->weight(std::get<1>(e)));
        }

        void lattice_score::operator()(
            feat_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            feat.class_param["[lattice] shared"].push_back(fst.fst1->weight(std::get<0>(e)));
        }

        void external_feature::operator()(
            feat_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            auto lm = *fst.fst2;
            auto lat = *fst.fst1;

            std::vector<double> vec;

            for (auto& p: lat.data->attrs[std::get<0>(e)]) {
                if (ebt::in(p.first, feature_keys)) {
                    vec.push_back(std::stod(p.second));
                }
            }

            if (vec.size() != feature_keys.size()) {
                std::cerr << "external features missing" << std::endl;
                exit(1);
            }

            std::string label_tuple = "[external] ";

            if (order == 0) {
                label_tuple += "shared";
            } else if (order == 1) {
                label_tuple += fst.output(e);
            } else if (order == 2) {
                auto const& lm = *fst.fst2;
                label_tuple += lm.data->history.at(std::get<1>(fst.tail(e))) + "_" + fst.output(e);
            } else {
                std::cerr << "order " << order << " not implemented" << std::endl;
                exit(1);
            }

            feat.class_param[label_tuple] = std::move(vec);
        }

    }

}
