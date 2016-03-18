#include "scrf/scrf_feat.h"
#include <cassert>

namespace scrf {

    composite_feature::composite_feature()
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
            std::unordered_map<std::string, int> const& label_dim)
            : frames(frames), label_dim(label_dim)
        {}
        
        void frame_feature::operator()(
            scrf::feat_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            auto& lat = *fst.fst1;
            int tail_time = std::min<int>(frames.size() - 1, lat.data->vertices.at(std::get<0>(fst.tail(e))).time);
            int head_time = std::min<int>(frames.size(), lat.data->vertices.at(std::get<0>(fst.head(e))).time);

            double sum = 0;
            int dim = label_dim.at(fst.output(e));
            for (int i = tail_time; i < head_time; ++i) {
                sum += frames[i][dim];
            }

            feat.class_vec[""].push_back(sum);
        }

    }

    namespace first_order {

        composite_feature::composite_feature()
        {}

        void composite_feature::operator()(
            param_t& feat, ilat::fst const& fst, int e) const
        {
            for (auto& f: features) {
                (*f)(feat, fst, e);
            }
        }
        
        la::vector<double>& lexicalize(feat_dim_alloc const& alloc,
            int order, param_t& feat, ilat::fst const& fst, int e)
        {
            int label_tuple = 0;

            if (order == 0) {
                // do nothing
            } else if (order == 1) {
                label_tuple = fst.output(e) + 1;
            } else {
                std::cerr << "order " << order << " not implemented" << std::endl;
                exit(1);
            }

            feat.class_vec.resize(alloc.labels.size() + 1);

            auto& g = feat.class_vec[label_tuple];
            g.resize(alloc.order_dim[order]);

            return g;
        }

        segment_feature::segment_feature(
            feat_dim_alloc& alloc,
            int order,
            std::shared_ptr<segfeat::la::feature> feat_func,
            std::vector<std::vector<real>> const& frames)
            : alloc(alloc), order(order), feat_func(feat_func), frames(frames)
        {
            dim = alloc.alloc(order, feat_func->dim(frames.front().size()));
        }

        void segment_feature::operator()(
            param_t& feat, ilat::fst const& fst, int e) const
        {
            la::vector<double>& g = lexicalize(alloc, order, feat, fst, e);

            (*feat_func)(dim, g, frames, fst.time(fst.tail(e)), fst.time(fst.head(e)));
        }

        namespace feature {

            lattice_score::lattice_score(feat_dim_alloc& alloc)
                : alloc(alloc)
            {
                dim = alloc.alloc(0, 1);
            }

            void lattice_score::operator()(
                param_t& feat, ilat::fst const& fst, int e) const
            {
                if (feat.class_vec.size() == 0) {
                    feat.class_vec.resize(1);
                    feat.class_vec[0].resize(alloc.order_dim[0]);
                }
                feat.class_vec[0](dim) = fst.weight(e);
            }

            external_feature::external_feature(feat_dim_alloc& alloc,
                    int order, std::vector<int> dims)
                : alloc(alloc), order(order), dims(dims)
            {
                dim = alloc.alloc(order, dims.size());
            }

            void external_feature::operator()(
                param_t& feat, ilat::fst const& fst, int e) const
            {
                auto& g = lexicalize(alloc, order, feat, fst, e);

                std::vector<double> const& f = fst.data->feats.at(e);

                for (int i = 0; i < dims.size(); ++i) {
                    g(dim + i) = f[dims[i]];
                }
            }

            frame_feature::frame_feature(feat_dim_alloc& alloc,
                    std::vector<std::vector<double>> const& frames,
                    std::vector<int> id_dim)
                : alloc(alloc), frames(frames), id_dim(id_dim)
            {
                dim = alloc.alloc(0, 1);
            }
            
            void frame_feature::operator()(
                param_t& feat, ilat::fst const& fst, int e) const
            {
                double sum = 0;

                int f_dim = id_dim[fst.output(e)];

                for (int i = fst.time(fst.tail(e)); i < fst.time(fst.head(e)); ++i) {
                    sum += frames[i][f_dim];
                }

                if (feat.class_vec.size() == 0) {
                    feat.class_vec.resize(1);
                    feat.class_vec[0].resize(alloc.order_dim[0]);
                }
                feat.class_vec[0](dim) = sum;
            }

        }
    }
}
