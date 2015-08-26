#ifndef FEAT_H
#define FEAT_H

#include "scrf/util.h"
#include "scrf/scrf.h"
#include "scrf/segfeat.h"
#include "scrf/nn.h"

namespace scrf {

    namespace feature {

        struct bias
            : public scrf_feature {

            bias();

            virtual std::string name() const override;

            virtual int size() const override;

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;

        };

        struct length_value
            : public scrf_feature {

            int max_seg;

            length_value(int max_seg);

            virtual int size() const override;

            virtual std::string name() const override;

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct length_indicator
            : public scrf_feature {

            int max_seg;

            length_indicator(int max_seg);

            virtual int size() const override;

            virtual std::string name() const override;

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct frame_avg
            : public scrf_feature {

            std::vector<std::vector<real>> const& inputs;
            int start_dim;
            int end_dim;

            mutable std::unordered_map<int, std::vector<real>> feat_cache;

            frame_avg(std::vector<std::vector<real>> const& inputs,
                int start_dim = -1, int end_dim = -1);

            virtual int size() const override;

            virtual std::string name() const override;

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct frame_samples
            : public scrf_feature {

            std::vector<std::vector<real>> const& inputs;
            int samples;
            int start_dim;
            int end_dim;

            frame_samples(std::vector<std::vector<real>> const& inputs, int samples,
                int start_dim = -1, int end_dim = -1);

            virtual int size() const override;

            virtual std::string name() const override;

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct left_boundary
            : public scrf_feature {

            std::vector<std::vector<real>> const& inputs;
            int start_dim;
            int end_dim;

            mutable std::unordered_map<int, std::vector<real>> feat_cache;

            left_boundary(std::vector<std::vector<real>> const& inputs,
                int start_dim = -1, int end_dim = -1);

            virtual int size() const override;

            virtual std::string name() const override;

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct right_boundary
            : public scrf_feature {

            std::vector<std::vector<real>> const& inputs;
            int start_dim;
            int end_dim;

            mutable std::unordered_map<int, std::vector<real>> feat_cache;

            right_boundary(std::vector<std::vector<real>> const& inputs,
                int start_dim = -1, int end_dim = -1);

            virtual int size() const override;

            virtual std::string name() const override;

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct lm_score
            : public scrf_feature {

            virtual int size() const override;

            virtual std::string name() const override;

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct lattice_score
            : public scrf_feature {

            virtual int size() const override;

            virtual std::string name() const override;

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;

        };

    }

    struct lexicalized_feature
        : public scrf_feature {

        lexicalized_feature(
            int order,
            std::shared_ptr<segfeat::feature> raw_feat_func,
            std::vector<std::vector<real>> const& frames);

        int order;
        std::shared_ptr<segfeat::feature> feat_func;
        std::vector<std::vector<real>> const& frames;

        virtual int size() const override;
        virtual std::string name() const override;

        virtual void operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const override;
    };

    composite_feature make_feature(
        std::vector<std::string> features,
        std::vector<std::vector<real>> const& inputs, int max_seg);

    composite_feature make_feature(
        std::vector<std::string> features,
        std::vector<std::vector<real>> const& inputs, int max_seg,
        std::vector<real> const& cm_mean, std::vector<real> const& cm_stddev,
        nn::nn_t const& nn);

    composite_feature make_feature2(
        std::vector<std::string> features,
        std::vector<std::vector<real>> const& frames);

}

#endif
