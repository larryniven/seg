#ifndef FEAT_H
#define FEAT_H

#include "scrf/util.h"
#include "scrf/scrf.h"

namespace segfeat {

    using feat_t = std::vector<real>;

    struct feature {

        ~feature();

        virtual void operator()(feat_t& feat,
            std::vector<std::vector<real>> const& frames,
            int start_time, int end_time) const = 0;

    };

    struct composite_feature
        : public feature {

        std::vector<std::shared_ptr<feature>> features;

        virtual void operator()(feat_t& feat,
            std::vector<std::vector<real>> const& frames,
            int start_time, int end_time) const override;

    };

    struct bias
        : public feature {

        virtual void operator()(feat_t& feat,
            std::vector<std::vector<real>> const& frames,
            int start_time, int end_time) const override;

    };

    struct length_indicator
        : public feature {

        length_indicator(int max_length);

        int max_length;

        virtual void operator()(feat_t& feat,
            std::vector<std::vector<real>> const& frames,
            int start_time, int end_time) const override;

    };

    void check_dim(std::vector<std::vector<real>> const& frames,
        int start_dim, int end_dim);

    struct frame_avg
        : public feature {

        frame_avg(int start_dim = -1, int end_dim = -1);

        int start_dim;
        int end_dim;

        virtual void operator()(feat_t& feat,
            std::vector<std::vector<real>> const& frames,
            int start_time, int end_time) const override;

    };

    struct frame_samples
        : public feature {

        frame_samples(int samples, int start_dim = -1, int end_dim = -1);

        int samples;
        int start_dim;
        int end_dim;

        virtual void operator()(feat_t& feat,
            std::vector<std::vector<real>> const& frames,
            int start_time, int end_time) const override;

    };

    struct left_boundary
        : public feature {

        left_boundary(int start_dim, int end_dim);

        int start_dim;
        int end_dim;

        virtual void operator()(feat_t& feat,
            std::vector<std::vector<real>> const& frames,
            int start_time, int end_time) const override;
    };

}

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

        struct tied_lattice_feature
            : public scrf_feature {

            tied_lattice_feature(std::vector<std::string> features);

            virtual int size() const override;

            virtual std::string name() const override;

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;

        private:
            mutable std::unordered_map<int, std::vector<real>> cache_;

            std::vector<std::string> features_;

        };

        struct lex_lattice_feature
            : public scrf_feature {

            lex_lattice_feature(std::vector<std::string> features);

            virtual int size() const override;

            virtual std::string name() const override;

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;

        private:
            mutable std::unordered_map<int, std::vector<real>> cache_;

            std::vector<std::string> features_;

        };

    }

}

#endif
