#include "scrf/experimental/segfeat.h"
#include <cassert>

namespace segfeat {

    feature::~feature()
    {}

    void bias::operator()(double *feat,
        std::vector<std::vector<double>> const& frames,
        int start_time, int end_time) const
    {
        feat[0] = 1;
    }

    int bias::dim(int frame_dim) const
    {
        return 1;
    }

    length_indicator::length_indicator(int max_length)
        : max_length(max_length)
    {}

    void length_indicator::operator()(double *feat,
        std::vector<std::vector<double>> const& frames,
        int start_time, int end_time) const
    {
        int d = end_time - start_time;

        if (d <= 1) {
            feat[0] = 1;
        } else if (d >= max_length) {
            feat[max_length - 1] = 1;
        } else {
            feat[d - 1] = 1;
        }
    }

    int length_indicator::dim(int frame_dim) const
    {
        return max_length;
    }

    length_separator::length_separator(int max_length)
        : max_length(max_length)
    {}

    void length_separator::operator()(double *feat,
        std::vector<std::vector<double>> const& frames,
        int start_time, int end_time) const
    {
        int d = end_time - start_time;

        for (int i = 0; i < max_length; ++i) {
            if (d >= i + 1) {
                feat[i] = 1;
            }
        }
    }

    int length_separator::dim(int frame_dim) const
    {
        return max_length;
    }

    void check_dim(std::vector<std::vector<double>> const& frames,
        int start_dim, int end_dim)
    {
        assert(0 <= start_dim && start_dim < frames.front().size());
        assert(0 <= end_dim && end_dim < frames.front().size());
    }

    frame_avg::frame_avg(std::vector<std::vector<double>> const& frames, int start_dim, int end_dim)
        : start_dim(start_dim), end_dim(end_dim)
    {
        assert(frames.size() >= 1);

        std::vector<double> prev;
        prev.resize(frames.front().size());

        accu.push_back(prev);

        for (int i = 0; i < frames.size(); ++i) {
            for (int d = 0; d < frames.front().size(); ++d) {
                prev[d] += frames[i][d];
            }

            accu.push_back(prev);
        }
    }

    void frame_avg::operator()(double *feat,
        std::vector<std::vector<double>> const& frames,
        int start_time, int end_time) const
    {
        assert(frames.size() >= 1);

        int capped_start_dim = (start_dim == -1 ? 0 : start_dim);
        int capped_end_dim = (end_dim == -1 ? frames.front().size() - 1 : end_dim);

        check_dim(frames, capped_start_dim, capped_end_dim);

        start_time = std::min<int>(start_time, frames.size() - 1);
        end_time = std::min<int>(end_time, frames.size());

        if (start_time >= end_time) {
            return;
        }

        int duration = end_time - start_time;

        auto& u = accu[end_time];
        auto& v = accu[start_time];

        for (int d = capped_start_dim; d <= capped_end_dim; ++d) {
            feat[d - capped_start_dim] = (u[d] - v[d]) / duration;
        }
    }

    void frame_avg::frame_grad(
        std::vector<std::vector<double>>& grad,
        double const *feat_grad,
        std::vector<std::vector<double>> const& frames,
        int start_time, int end_time) const
    {
        assert(frames.size() >= 1);

        int capped_start_dim = (start_dim == -1 ? 0 : start_dim);
        int capped_end_dim = (end_dim == -1 ? frames.front().size() - 1 : end_dim);

        check_dim(frames, capped_start_dim, capped_end_dim);

        start_time = std::min<int>(start_time, frames.size() - 1);
        end_time = std::min<int>(end_time, frames.size());

        if (start_time >= end_time) {
            return;
        }

        int duration = end_time - start_time;

        auto& u = accu[end_time];
        auto& v = accu[start_time];

        for (int t = start_time; t < end_time; ++t) {
            for (int d = capped_start_dim; d <= capped_end_dim; ++d) {
                grad[t][d] += feat_grad[d - capped_start_dim] / duration;
            }
        }
    }

    int frame_avg::dim(int frame_dim) const
    {
        return frame_dim;
    }

    frame_samples::frame_samples(int samples, int start_dim, int end_dim)
        : samples(samples), start_dim(start_dim), end_dim(end_dim)
    {}

    void frame_samples::operator()(double *feat,
        std::vector<std::vector<double>> const& frames,
        int start_time, int end_time) const
    {
        assert(frames.size() >= 1);

        int capped_start_dim = (start_dim == -1 ? 0 : start_dim);
        int capped_end_dim = (end_dim == -1 ? frames.front().size() - 1 : end_dim);

        check_dim(frames, capped_start_dim, capped_end_dim);

        double span = (end_time - start_time) / samples;
        int length = capped_end_dim - capped_start_dim + 1;

        if (start_time >= end_time) {
            return;
        }

        for (int i = 0; i < samples; ++i) {
            auto& u = frames.at(std::min<int>(
                std::floor(start_time + (i + 0.5) * span), frames.size() - 1));

            int base = i * length - capped_start_dim;
            for (int d = capped_start_dim; d <= capped_end_dim; ++d) {
                feat[base + d] = u[d];
            }
        }
    }

    void frame_samples::frame_grad(
        std::vector<std::vector<double>>& grad,
        double const *feat_grad,
        std::vector<std::vector<double>> const& frames,
        int start_time, int end_time) const
    {
        assert(frames.size() >= 1);

        int capped_start_dim = (start_dim == -1 ? 0 : start_dim);
        int capped_end_dim = (end_dim == -1 ? frames.front().size() - 1 : end_dim);

        check_dim(frames, capped_start_dim, capped_end_dim);

        double span = (end_time - start_time) / samples;
        int length = capped_end_dim - capped_start_dim + 1;

        if (start_time >= end_time) {
            return;
        }

        for (int i = 0; i < samples; ++i) {
            auto& u = grad.at(std::min<int>(
                std::floor(start_time + (i + 0.5) * span), frames.size() - 1));

            int base = i * length - capped_start_dim;
            for (int d = capped_start_dim; d <= capped_end_dim; ++d) {
                u[d] += feat_grad[base + d];
            }
        }
    }

    int frame_samples::dim(int frame_dim) const
    {
        return samples * frame_dim;
    }

    left_boundary::left_boundary(int start_dim, int end_dim)
        : start_dim(start_dim), end_dim(end_dim)
    {}

    void left_boundary::operator()(double *feat,
        std::vector<std::vector<double>> const& frames,
        int start_time, int end_time) const
    {
        assert(frames.size() >= 1);

        int capped_start_dim = (start_dim == -1 ? 0 : start_dim);
        int capped_end_dim = (end_dim == -1 ? frames.front().size() - 1 : end_dim);

        check_dim(frames, capped_start_dim, capped_end_dim);

        int length = capped_end_dim - capped_start_dim + 1;

        if (start_time >= end_time) {
            return;
        }

        for (int i = 0; i < 3; ++i) {
            auto& tail_u = frames.at(std::min<int>(frames.size() - 1,
                std::max<int>(start_time - i, 0)));

            int base = i * length - capped_start_dim;
            for (int d = capped_start_dim; d <= capped_end_dim; ++d) {
                feat[base + d] = tail_u[d];
            }
        }
    }

    void left_boundary::frame_grad(
        std::vector<std::vector<double>>& grad,
        double const *feat_grad,
        std::vector<std::vector<double>> const& frames,
        int start_time, int end_time) const
    {
        assert(frames.size() >= 1);

        int capped_start_dim = (start_dim == -1 ? 0 : start_dim);
        int capped_end_dim = (end_dim == -1 ? frames.front().size() - 1 : end_dim);

        check_dim(frames, capped_start_dim, capped_end_dim);

        int length = capped_end_dim - capped_start_dim + 1;

        if (start_time >= end_time) {
            return;
        }

        for (int i = 0; i < 3; ++i) {
            auto& tail_u = grad.at(std::min<int>(frames.size() - 1,
                std::max<int>(start_time - i, 0)));

            int base = i * length - capped_start_dim;
            for (int d = capped_start_dim; d <= capped_end_dim; ++d) {
                tail_u[d] += feat_grad[base + d];
            }
        }
    }

    int left_boundary::dim(int frame_dim) const
    {
        return 3 * frame_dim;
    }

    right_boundary::right_boundary(int start_dim, int end_dim)
        : start_dim(start_dim), end_dim(end_dim)
    {}

    void right_boundary::operator()(double *feat,
        std::vector<std::vector<double>> const& frames,
        int start_time, int end_time) const
    {
        assert(frames.size() >= 1);

        int capped_start_dim = (start_dim == -1 ? 0 : start_dim);
        int capped_end_dim = (end_dim == -1 ? frames.front().size() - 1 : end_dim);

        check_dim(frames, capped_start_dim, capped_end_dim);

        int length = capped_end_dim - capped_start_dim + 1;

        if (start_time >= end_time) {
            return;
        }

        for (int i = 0; i < 3; ++i) {
            auto& head_u = frames.at(std::min<int>(frames.size() - 1,
                std::max<int>(end_time + i, 0)));

            int base = i * length - capped_start_dim;
            for (int d = capped_start_dim; d <= capped_end_dim; ++d) {
                feat[base + d] = head_u[d];
            }
        }
    }

    void right_boundary::frame_grad(
        std::vector<std::vector<double>>& grad,
        double const *feat_grad,
        std::vector<std::vector<double>> const& frames,
        int start_time, int end_time) const
    {
        assert(frames.size() >= 1);

        int capped_start_dim = (start_dim == -1 ? 0 : start_dim);
        int capped_end_dim = (end_dim == -1 ? frames.front().size() - 1 : end_dim);

        check_dim(frames, capped_start_dim, capped_end_dim);

        int length = capped_end_dim - capped_start_dim + 1;

        if (start_time >= end_time) {
            return;
        }

        for (int i = 0; i < 3; ++i) {
            auto& head_u = grad.at(std::min<int>(frames.size() - 1,
                std::max<int>(end_time + i, 0)));

            int base = i * length - capped_start_dim;
            for (int d = capped_start_dim; d <= capped_end_dim; ++d) {
                head_u[d] += feat_grad[base + d];
            }
        }
    }

    int right_boundary::dim(int frame_dim) const
    {
        return 3 * frame_dim;
    }

}

