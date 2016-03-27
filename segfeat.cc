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
        feat.resize(orig_size + max_length);

        int d = end_time - start_time;

        if (1 <= d) {
            feat[orig_size] = 1;
        } else if (d >= max_length) {
            feat[orig_size + max_length - 1] = 1;
        } else {
            feat[orig_size + d - 1] = 1;
        }
    }

    void check_dim(std::vector<std::vector<real>> const& frames,
        int start_dim, int end_dim)
    {
        assert(0 <= start_dim && start_dim < frames.front().size());
        assert(0 <= end_dim && end_dim < frames.front().size());
    }

    frame_avg::frame_avg(std::vector<std::vector<real>> const& frames, int start_dim, int end_dim)
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

        start_time = std::min<int>(start_time, frames.size());
        end_time = std::min<int>(end_time, frames.size());

        auto& u = accu[end_time];
        auto& v = accu[start_time];
        int duration = end_time - start_time;

        for (int d = capped_start_dim; d <= capped_end_dim; ++d) {
            result[d - capped_start_dim] = (u[d] - v[d]) / duration;
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
                std::floor(start_time + (i + 0.5) * span), frames.size() - 1));

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

    namespace la {

        feature::~feature()
        {}

        void bias::operator()(int dim, feat_t& feat,
            std::vector<std::vector<real>> const& frames,
            int start_time, int end_time) const
        {
            feat(dim) = 1;
        }

        int bias::dim(int frame_dim) const
        {
            return 1;
        }

        length_indicator::length_indicator(int max_length)
            : max_length(max_length)
        {}

        void length_indicator::operator()(int dim, feat_t& feat,
            std::vector<std::vector<real>> const& frames,
            int start_time, int end_time) const
        {
            int d = end_time - start_time;

            if (1 <= d) {
                feat(dim) = 1;
            } else if (d >= max_length) {
                feat(dim + max_length - 1) = 1;
            } else {
                feat(dim + d - 1) = 1;
            }
        }

        int length_indicator::dim(int frame_dim) const
        {
            return max_length;
        }

        void check_dim(std::vector<std::vector<real>> const& frames,
            int start_dim, int end_dim)
        {
            assert(0 <= start_dim && start_dim < frames.front().size());
            assert(0 <= end_dim && end_dim < frames.front().size());
        }

        frame_avg::frame_avg(std::vector<std::vector<real>> const& frames, int start_dim, int end_dim)
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

        void frame_avg::operator()(int dim, feat_t& feat,
            std::vector<std::vector<real>> const& frames,
            int start_time, int end_time) const
        {
            assert(frames.size() >= 1);

            int capped_start_dim = (start_dim == -1 ? 0 : start_dim);
            int capped_end_dim = (end_dim == -1 ? frames.front().size() - 1 : end_dim);

            check_dim(frames, capped_start_dim, capped_end_dim);

            start_time = std::min<int>(start_time, frames.size() - 1);
            end_time = std::min<int>(end_time, frames.size());

            int duration = end_time - start_time;

            if (duration <= 0) {
                return;
            }

            auto& u = accu[end_time];
            auto& v = accu[start_time];

            int base = dim - capped_start_dim;
            for (int d = capped_start_dim; d <= capped_end_dim; ++d) {
                feat(base + d) = (u[d] - v[d]) / duration;
            }
        }

        int frame_avg::dim(int frame_dim) const
        {
            return frame_dim;
        }

        frame_samples::frame_samples(int samples, int start_dim, int end_dim)
            : samples(samples), start_dim(start_dim), end_dim(end_dim)
        {}

        void frame_samples::operator()(int dim, feat_t& feat,
            std::vector<std::vector<real>> const& frames,
            int start_time, int end_time) const
        {
            assert(frames.size() >= 1);

            int capped_start_dim = (start_dim == -1 ? 0 : start_dim);
            int capped_end_dim = (end_dim == -1 ? frames.front().size() - 1 : end_dim);

            check_dim(frames, capped_start_dim, capped_end_dim);

            real span = (end_time - start_time) / samples;
            int length = capped_end_dim - capped_start_dim + 1;

            if (start_time >= end_time) {
                return;
            }

            for (int i = 0; i < samples; ++i) {
                auto& u = frames.at(std::min<int>(
                    std::floor(start_time + (i + 0.5) * span), frames.size() - 1));

                int base = dim + i * length - capped_start_dim;
                for (int d = capped_start_dim; d <= capped_end_dim; ++d) {
                    feat(base + d) = u[d];
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

        void left_boundary::operator()(int dim, feat_t& feat,
            std::vector<std::vector<real>> const& frames,
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

                int base = dim + i * length - capped_start_dim;
                for (int d = capped_start_dim; d <= capped_end_dim; ++d) {
                    feat(base + d) = tail_u[d];
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

        void right_boundary::operator()(int dim, feat_t& feat,
            std::vector<std::vector<real>> const& frames,
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

                int base = dim + i * length - capped_start_dim;
                for (int d = capped_start_dim; d <= capped_end_dim; ++d) {
                    feat(base + d) = head_u[d];
                }
            }
        }

        int right_boundary::dim(int frame_dim) const
        {
            return 3 * frame_dim;
        }

    }
}

