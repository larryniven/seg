#include "scrf/segfeat.h"
#include <cassert>

namespace segfeat {

    feature::~feature()
    {}

    void composite_feature::operator()(feat_t& feat,
        std::vector<std::vector<real>> const& frames,
        int start_time, int end_time) const
    {
        for (auto& f: features) {
            (*f)(feat, frames, start_time, end_time);
        }
    }

    void bias::operator()(feat_t& feat,
        std::vector<std::vector<real>> const& frames,
        int start_time, int end_time) const
    {
        feat.push_back(1);
    }

    length_indicator::length_indicator(int max_length)
        : max_length(max_length)
    {}

    void length_indicator::operator()(feat_t& feat,
        std::vector<std::vector<real>> const& frames,
        int start_time, int end_time) const
    {
        int orig_size = feat.size();
        feat.resize(orig_size + max_length + 1);

        if (0 <= end_time - start_time && end_time - start_time <= max_length) {
            feat[orig_size + end_time - start_time] = 1;
        }
    }

    void check_dim(std::vector<std::vector<real>> const& frames,
        int start_dim, int end_dim)
    {
        assert(0 <= start_dim && start_dim < frames.front().size());
        assert(0 <= end_dim && end_dim < frames.front().size());
    }

    frame_avg::frame_avg(int start_dim, int end_dim)
        : start_dim(start_dim), end_dim(end_dim)
    {}

    void frame_avg::operator()(feat_t& feat,
        std::vector<std::vector<real>> const& frames,
        int start_time, int end_time) const
    {
        assert(frames.size() >= 1);

        int capped_start_dim = (start_dim == -1 ? 0 : start_dim);
        int capped_end_dim = (end_dim == -1 ? frames.front().size() - 1 : end_dim);

        check_dim(frames, capped_start_dim, capped_end_dim);

        std::vector<real> result;
        result.resize(capped_end_dim - capped_start_dim + 1);

        if (start_time >= end_time) {
            feat.insert(feat.end(), result.begin(), result.end());
            return;
        }

        for (int i = start_time; i < end_time; ++i) {
            int capped_i = std::min<int>(std::max<int>(i, 0), frames.size() - 1);
            auto& v = frames[capped_i];

            for (int d = capped_start_dim; d <= capped_end_dim; ++d) {
                result[d - capped_start_dim] += v[d];
            }
        }

        for (int d = capped_start_dim; d <= capped_end_dim; ++d) {
            result[d - capped_start_dim] /= (end_time - start_time);
        }

        feat.insert(feat.end(), result.begin(), result.end());
    }

    frame_samples::frame_samples(int samples, int start_dim, int end_dim)
        : samples(samples), start_dim(start_dim), end_dim(end_dim)
    {}

    void frame_samples::operator()(feat_t& feat,
        std::vector<std::vector<real>> const& frames,
        int start_time, int end_time) const
    {
        assert(frames.size() >= 1);

        int capped_start_dim = (start_dim == -1 ? 0 : start_dim);
        int capped_end_dim = (end_dim == -1 ? frames.front().size() - 1 : end_dim);

        check_dim(frames, capped_start_dim, capped_end_dim);

        real span = (end_time - start_time) / samples;
        int length = capped_end_dim - capped_start_dim + 1;

        std::vector<real> result;
        result.resize(samples * length);

        if (start_time >= end_time) {
            feat.insert(feat.end(), result.begin(), result.end());
            return;
        }

        for (int i = 0; i < samples; ++i) {
            auto& u = frames.at(std::min<int>(
                std::floor(end_time + (i + 0.5) * span), frames.size() - 1));

            for (int d = capped_start_dim; d <= capped_end_dim; ++d) {
                result[i * length + d - capped_start_dim] = u[d];
            }
        }

        feat.insert(feat.end(), result.begin(), result.end());
    }

    left_boundary::left_boundary(int start_dim, int end_dim)
        : start_dim(start_dim), end_dim(end_dim)
    {}

    void left_boundary::operator()(feat_t& feat,
        std::vector<std::vector<real>> const& frames,
        int start_time, int end_time) const
    {
        assert(frames.size() >= 1);

        int capped_start_dim = (start_dim == -1 ? 0 : start_dim);
        int capped_end_dim = (end_dim == -1 ? frames.front().size() - 1 : end_dim);

        check_dim(frames, capped_start_dim, capped_end_dim);

        std::vector<double> result;
        int length = capped_end_dim - capped_start_dim + 1;
        result.resize(3 * length);

        if (start_time >= end_time) {
            feat.insert(feat.begin(), result.begin(), result.end());
            return;
        }

        for (int i = 0; i < 3; ++i) {
            auto& tail_u = frames.at(std::min<int>(frames.size() - 1,
                std::max<int>(start_time - i, 0)));

            for (int d = capped_start_dim; d <= capped_end_dim; ++d) {
                result[i * length + d - capped_start_dim] = tail_u[d];
            }
        }

        feat.insert(feat.begin(), result.begin(), result.end());
    }

    right_boundary::right_boundary(int start_dim, int end_dim)
        : start_dim(start_dim), end_dim(end_dim)
    {}

    void right_boundary::operator()(feat_t& feat,
        std::vector<std::vector<real>> const& frames,
        int start_time, int end_time) const
    {
        assert(frames.size() >= 1);

        int capped_start_dim = (start_dim == -1 ? 0 : start_dim);
        int capped_end_dim = (end_dim == -1 ? frames.front().size() - 1 : end_dim);

        check_dim(frames, capped_start_dim, capped_end_dim);

        std::vector<double> result;
        int length = capped_end_dim - capped_start_dim + 1;
        result.resize(3 * length);

        if (start_time >= end_time) {
            feat.insert(feat.begin(), result.begin(), result.end());
            return;
        }

        for (int i = 0; i < 3; ++i) {
            auto& head_u = frames.at(std::min<int>(frames.size() - 1,
                std::max<int>(end_time + i, 0)));

            for (int d = capped_start_dim; d <= capped_end_dim; ++d) {
                result[i * length + d - capped_start_dim] = head_u[d];
            }
        }

        feat.insert(feat.begin(), result.begin(), result.end());
    }
}

