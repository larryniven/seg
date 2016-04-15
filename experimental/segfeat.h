#ifndef SEG_FEAT_H
#define SEG_FEAT_H

#include <vector>
#include <cmath>
#include <memory>
#include "la/la.h"

namespace segfeat {

    struct feature {

        ~feature();

        virtual int dim(int frame_dim) const = 0;

        virtual void operator()(double *feat,
            std::vector<std::vector<double>> const& frames,
            int start_time, int end_time) const = 0;

    };

    struct with_frame_grad {

        virtual void frame_grad(
            std::vector<std::vector<double>>& grad,
            double const *param,
            std::vector<std::vector<double>> const& frames,
            int start_time, int end_time) const = 0;

    };

    struct feature_with_frame_grad
        : public feature
        , public with_frame_grad {

    };

    struct bias
        : public feature {

        virtual int dim(int frame_dim) const override;

        virtual void operator()(double *fst,
            std::vector<std::vector<double>> const& frames,
            int start_time, int end_time) const override;

    };

    struct length_indicator
        : public feature {

        length_indicator(int max_length);

        int max_length;

        virtual int dim(int frame_dim) const override;

        virtual void operator()(double *feat,
            std::vector<std::vector<double>> const& frames,
            int start_time, int end_time) const override;

    };

    struct length_separator
        : public feature {

        length_separator(int max_length);

        int max_length;

        virtual int dim(int frame_dim) const override;

        virtual void operator()(double *feat,
            std::vector<std::vector<double>> const& frames,
            int start_time, int end_time) const override;

    };

    void check_dim(std::vector<std::vector<double>> const& frames,
        int start_dim, int end_dim);

    struct frame_avg
        : public feature_with_frame_grad {

        frame_avg(std::vector<std::vector<double>> const& frames, int start_dim = -1, int end_dim = -1);

        int start_dim;
        int end_dim;

        std::vector<std::vector<double>> accu;

        virtual int dim(int frame_dim) const override;

        virtual void operator()(double *feat,
            std::vector<std::vector<double>> const& frames,
            int start_time, int end_time) const override;

        virtual void frame_grad(
            std::vector<std::vector<double>>& grad,
            double const *param,
            std::vector<std::vector<double>> const& frames,
            int start_time, int end_time) const override;

    };

    struct frame_samples
        : public feature {

        frame_samples(int samples, int start_dim = -1, int end_dim = -1);

        int samples;
        int start_dim;
        int end_dim;

        virtual int dim(int frame_dim) const override;

        virtual void operator()(double *feat,
            std::vector<std::vector<double>> const& frames,
            int start_time, int end_time) const override;

    };

    struct left_boundary
        : public feature {

        left_boundary(int start_dim, int end_dim);

        int start_dim;
        int end_dim;

        virtual int dim(int frame_dim) const override;

        virtual void operator()(double *feat,
            std::vector<std::vector<double>> const& frames,
            int start_time, int end_time) const override;
    };

    struct right_boundary
        : public feature {

        right_boundary(int start_dim, int end_dim);

        int start_dim;
        int end_dim;

        virtual int dim(int frame_dim) const override;

        virtual void operator()(double *feat,
            std::vector<std::vector<double>> const& frames,
            int start_time, int end_time) const override;
    };

}

#endif
