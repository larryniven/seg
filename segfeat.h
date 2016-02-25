#ifndef SEG_FEAT_H
#define SEG_FEAT_H

#include "scrf/util.h"
#include <vector>
#include <cmath>
#include <memory>

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

        frame_avg(std::vector<std::vector<real>> const& frames, int start_dim = -1, int end_dim = -1);

        int start_dim;
        int end_dim;

        std::vector<std::vector<real>> accu;

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

    struct right_boundary
        : public feature {

        right_boundary(int start_dim, int end_dim);

        int start_dim;
        int end_dim;

        virtual void operator()(feat_t& feat,
            std::vector<std::vector<real>> const& frames,
            int start_time, int end_time) const override;
    };

}

#endif
