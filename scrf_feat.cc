#include "scrf/scrf_feat.h"
#include <cassert>

namespace scrf {

    composite_feature::composite_feature()
    {}

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
        
    std::vector<double>& lexicalize(int order, feat_t& feat,
        fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e)
    {
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

        auto& g = feat.class_vec[label_tuple];
        g.reserve(1000);

        return g;
    }

    segment_feature::segment_feature(
        int order,
        std::shared_ptr<segfeat::feature> feat_func,
        std::vector<std::vector<real>> const& frames)
        : order(order), feat_func(feat_func), frames(frames)
    {}

    void segment_feature::operator()(
        feat_t& feat,
        fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        std::vector<double>& g = lexicalize(order, feat, fst, e);

        lattice::fst const& lat = *fst.fst1;
        int tail_time = lat.data->vertices.at(std::get<0>(fst.tail(e))).time;
        int head_time = lat.data->vertices.at(std::get<0>(fst.head(e))).time;

        (*feat_func)(g, frames, tail_time, head_time);
    }

    namespace feature {

        void lm_score::operator()(
            feat_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            feat.class_vec[""].push_back(fst.fst2->weight(std::get<1>(e)));
        }

        void lattice_score::operator()(
            feat_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            feat.class_vec[""].push_back(fst.fst1->weight(std::get<0>(e)));
        }

        external_feature::external_feature(int order, std::vector<int> dims)
            : order(order), dims(dims)
        {}

        void external_feature::operator()(
            feat_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            auto& g = lexicalize(order, feat, fst, e);

            lattice::fst& lat = *fst.fst1;

            std::vector<double> const& f = lat.data->feats[std::get<0>(e)];

            if (fst.input(e) == "<eps>") {
                for (auto i: dims) {
                    g.push_back(0);
                }
            } else {
                for (auto i: dims) {
                    g.push_back(f[i]);
                }
            }
        }

        frame_feature::frame_feature(std::vector<std::vector<double>> const& frames,
            std::unordered_map<std::string, int> const& phone_id)
            : frames(frames), phone_id(phone_id)
        {}
        
        void frame_feature::operator()(
            scrf::feat_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            auto& lat = *fst.fst1;
            int tail_time = lat.data->vertices.at(std::get<0>(fst.tail(e))).time;
            int head_time = lat.data->vertices.at(std::get<0>(fst.head(e))).time;

            double sum = 0;
            for (int i = tail_time; i < head_time; ++i) {
                sum += frames[i][phone_id.at(fst.output(e))];
            }

            feat.class_vec[""].push_back(sum);
        }

    }

}
