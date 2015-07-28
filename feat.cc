#include "scrf/feat.h"
#include <cassert>

namespace segfeat {

    feature::~feature()
    {}

    void bias::operator()(feat_t& feat,
        std::vector<std::vector<real>> const& frames,
        int start_time, int end_time) const
    {
        feat["bias"] = std::vector<real> { 1 };
    }

    length_indicator::length_indicator(int max_length)
        : max_length(max_length)
    {}

    void length_indicator::operator()(feat_t& feat,
        std::vector<std::vector<real>> const& frames,
        int start_time, int end_time) const
    {
        std::vector<real> len;

        len.resize(max_length);

        if (start_time < end_time && end_time - start_time < max_length) {
            len[end_time - start_time - 1] = 1;
        }

        feat["length"] = std::move(len);
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

        feat["frame-avg"].resize(capped_end_dim - capped_start_dim + 1);
        std::vector<real>& result = feat["frame-avg"];

        if (start_time >= end_time) {
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

        std::vector<real>& result = feat["frame-samples"];

        if (start_time >= end_time) {
            return;
        }

        for (int i = 0; i < samples; ++i) {
            auto& u = frames.at(std::min<int>(
                std::floor(end_time + (i + 0.5) * span), frames.size() - 1));
            result.insert(result.end(), u.begin() + capped_start_dim,
                u.begin() + capped_end_dim + 1);
        }
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

        feat["left-boundary"].resize(3 * (capped_end_dim - capped_start_dim + 1));
        auto& result = feat["left-boundary"];

        if (start_time >= end_time) {
            return;
        }

        for (int i = 0; i < 3; ++i) {
            auto& tail_u = frames.at(std::min<int>(frames.size() - 1, std::max<int>(end_time - i, 0)));
            result.insert(result.end(), tail_u.begin() + capped_start_dim, tail_u.begin() + capped_end_dim + 1);
        }
    }
}

namespace scrf {

    namespace feature {

        bias::bias()
        {}

        int bias::size() const
        {
            return 2;
        }

        std::string bias::name() const
        {
            return "bias";
        }

        void bias::operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            feat.class_param["[label] " + fst.output(e)].push_back(1);
            feat.class_param["[label] shared"].push_back(1);
        }

        length_value::length_value(int max_seg)
            : max_seg(max_seg)
        {}

        int length_value::size() const
        {
            return 1;
        }

        std::string length_value::name() const
        {
            return "length-value";
        }

        void length_value::operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            auto const& lat = *(fst.fst1);
            int tail = lat.tail(std::get<0>(e));
            int head = lat.head(std::get<0>(e));

            int tail_time = lat.data->vertices.at(tail).time;
            int head_time = lat.data->vertices.at(head).time;

            auto& v = feat.class_param["[lattice] " + fst.output(e)];

            v.push_back(head_time - tail_time);
            v.push_back(std::pow(head_time - tail_time, 2));
        }

        length_indicator::length_indicator(int max_seg)
            : max_seg(max_seg)
        {}

        int length_indicator::size() const
        {
            return max_seg + 1;
        }

        std::string length_indicator::name() const
        {
            return "length-indicator";
        }

        void length_indicator::operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            auto const& lat = *(fst.fst1);
            int tail = lat.tail(std::get<0>(e));
            int head = lat.head(std::get<0>(e));

            int tail_time = lat.data->vertices.at(tail).time;
            int head_time = lat.data->vertices.at(head).time;

            auto& v = feat.class_param["[lattice] " + fst.output(e)];
            int size = v.size();
            v.resize(size + max_seg + 1);

            if (fst.output(e) != "<s>" && fst.output(e) != "</s>" && fst.output(e) != "sil") {
                v.at(size + std::min(head_time - tail_time, max_seg)) = 1;
            }
        }

        frame_avg::frame_avg(std::vector<std::vector<real>> const& inputs,
            int start_dim, int end_dim)
            : inputs(inputs), start_dim(start_dim), end_dim(end_dim)
        {
            if (this->start_dim == -1) {
                this->start_dim = 0;
            }
            if (this->end_dim == -1) {
                this->end_dim = inputs.front().size() - 1;
            }
        }

        int frame_avg::size() const
        {
            return end_dim - start_dim + 1;
        }

        std::string frame_avg::name() const
        {
            return "frame-avg";
        }

        void frame_avg::operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            if (ebt::in(std::get<0>(e), feat_cache)) {
                auto& u = feat_cache.at(std::get<0>(e));
                auto& v = feat.class_param["[lattice] " + fst.output(e)];
                v.insert(v.end(), u.begin(), u.end());
                return;
            }

            auto const& lat = *(fst.fst1);
            int tail = lat.tail(std::get<0>(e));
            int head = lat.head(std::get<0>(e));

            int tail_time = std::min<int>(inputs.size() - 1, lat.data->vertices.at(tail).time);
            int head_time = std::min<int>(inputs.size(), lat.data->vertices.at(head).time);

            std::vector<real> avg;
            avg.resize(end_dim - start_dim + 1);

            if (tail_time < head_time) {
                for (int i = tail_time; i < head_time; ++i) {
                    auto const& v = inputs.at(i);

                    for (int j = start_dim; j < end_dim + 1; ++j) {
                        avg[j - start_dim] += v.at(j);
                    }
                }

                for (int j = 0; j < avg.size(); ++j) {
                    avg[j] /= real(head_time - tail_time);
                }
            }

            auto& v = feat.class_param["[lattice] " + fst.output(e)];

            v.insert(v.end(), avg.begin(), avg.end());

            feat_cache[std::get<0>(e)] = std::move(avg);
        }

        frame_samples::frame_samples(std::vector<std::vector<real>> const& inputs,
            int samples, int start_dim, int end_dim)
            : inputs(inputs), samples(samples), start_dim(start_dim), end_dim(end_dim)
        {
            if (this->start_dim == -1) {
                this->start_dim = 0;
            }
            if (this->end_dim == -1) {
                this->end_dim = inputs.front().size() - 1;
            }
        }

        int frame_samples::size() const
        {
            return samples * (end_dim - start_dim + 1);
        }

        std::string frame_samples::name() const
        {
            return "frame-samples";
        }

        void frame_samples::operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            auto const& lat = *(fst.fst1);
            int tail = lat.tail(std::get<0>(e));
            int head = lat.head(std::get<0>(e));

            int tail_time = lat.data->vertices.at(tail).time;
            int head_time = lat.data->vertices.at(head).time;

            real span = (head_time - tail_time) / samples;

            auto& v = feat.class_param["[lattice] " + fst.output(e)];
            for (int i = 0; i < samples; ++i) {
                auto& u = inputs.at(std::min<int>(std::floor(tail_time + (i + 0.5) * span), inputs.size() - 1));
                v.insert(v.end(), u.begin() + start_dim, u.begin() + end_dim + 1);
            }
        }

        left_boundary::left_boundary(std::vector<std::vector<real>> const& inputs,
            int start_dim, int end_dim)
            : inputs(inputs), start_dim(start_dim), end_dim(end_dim)
        {
            if (this->start_dim == -1) {
                this->start_dim = 0;
            }
            if (this->end_dim == -1) {
                this->end_dim = inputs.front().size() - 1;
            }
        }

        int left_boundary::size() const
        {
            return 3 * (end_dim - start_dim + 1);
        }

        std::string left_boundary::name() const
        {
            return "left-boundary";
        }

        void left_boundary::operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            auto const& lm = *(fst.fst2);

            int lm_tail = lm.tail(std::get<1>(fst.tail(e)));

            std::string lex = lm.data->history.at(lm_tail) + "_" + fst.output(e);

            auto const& lat = *(fst.fst1);
            int tail = lat.tail(std::get<0>(e));
            int head = lat.head(std::get<0>(e));

            int tail_time = lat.data->vertices.at(tail).time;

            if (ebt::in(tail_time, feat_cache)) {
                auto& u = feat_cache.at(tail_time);
                auto& v = feat.class_param["[lattice] " + lex];
                v.insert(v.end(), u.begin(), u.end());
                return;
            }

            auto& v = feat.class_param["[lattice] " + lex];

            std::vector<real> f;
            for (int i = 0; i < 3; ++i) {
                auto& tail_u = inputs.at(std::min<int>(inputs.size() - 1, std::max<int>(tail_time - i, 0)));
                f.insert(f.end(), tail_u.begin() + start_dim, tail_u.begin() + end_dim + 1);
            }
            v.insert(v.end(), f.begin(), f.end());

            feat_cache[tail_time] = std::move(f);
        }

        right_boundary::right_boundary(std::vector<std::vector<real>> const& inputs,
            int start_dim, int end_dim)
            : inputs(inputs), start_dim(start_dim), end_dim(end_dim)
        {
            if (this->start_dim == -1) {
                this->start_dim = 0;
            }
            if (this->end_dim == -1) {
                this->end_dim = inputs.front().size() - 1;
            }
        }

        int right_boundary::size() const
        {
            return 3 * (end_dim - start_dim + 1);
        }

        std::string right_boundary::name() const
        {
            return "right-boundary";
        }

        void right_boundary::operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            auto const& lm = *(fst.fst2);

            int lm_tail = lm.tail(std::get<1>(fst.tail(e)));

            std::string lex = lm.data->history.at(lm_tail) + "_" + fst.output(e);

            auto const& lat = *(fst.fst1);
            int tail = lat.tail(std::get<0>(e));
            int head = lat.head(std::get<0>(e));

            int head_time = lat.data->vertices.at(head).time;

            if (ebt::in(head_time, feat_cache)) {
                auto& u = feat_cache.at(head_time);
                auto& v = feat.class_param["[lattice] " + lex];
                v.insert(v.end(), u.begin(), u.end());
                return;
            }

            auto& v = feat.class_param["[lattice] " + lex];

            std::vector<real> f;
            for (int i = 0; i < 3; ++i) {
                auto& tail_u = inputs.at(std::min<int>(head_time + i, inputs.size() - 1));
                f.insert(f.end(), tail_u.begin() + start_dim, tail_u.begin() + end_dim + 1);
            }
            v.insert(v.end(), f.begin(), f.end());

            feat_cache[head_time] = std::move(f);
        }

        int lm_score::size() const
        {
            return 1;
        }

        std::string lm_score::name() const
        {
            return "lm-score";
        }

        void lm_score::operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            feat.class_param["[lm] shared"].push_back(fst.fst2->weight(std::get<1>(e)));
        }

        int lattice_score::size() const
        {
            return 1;
        }

        std::string lattice_score::name() const
        {
            return "lattice-score";
        }

        void lattice_score::operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            feat.class_param["[lattice] shared"].push_back(fst.fst1->weight(std::get<0>(e)));
        }

        lex_lattice_feature::lex_lattice_feature(std::vector<std::string> features)
            : features_(features)
        {}

        int lex_lattice_feature::size() const
        {
            return features_.size();
        }

        std::string lex_lattice_feature::name() const
        {
            return "lex-lattice-feature";
        }

        void lex_lattice_feature::operator()(
            param_t& param,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            if (features_.size() == 0) {
                return;
            }

            if (ebt::in(std::get<0>(e), cache_)) {
                std::vector<real>& v = cache_.at(std::get<0>(e));
                auto& u = param.class_param["[lattice] " + fst.output(e)];
                u.insert(u.end(), v.begin(), v.end());
                return;
            }

            std::vector<real> result;

            auto& feat = fst.fst1->data->features.at(std::get<0>(e));

            for (auto& f: features_) {
                if (ebt::startswith(f, "lex@")) {
                    result.push_back(feat(f.substr(4)));
                } else {
                    std::cout << "Features " << f << " are expected to be lexicalized." << std::endl;
                    exit(1);
                }
            }

            cache_[std::get<0>(e)] = result;

            auto& u = param.class_param["[lattice] " + fst.output(e)];
            u.insert(u.end(), result.begin(), result.end());
        }

        tied_lattice_feature::tied_lattice_feature(std::vector<std::string> features)
            : features_(features)
        {}

        int tied_lattice_feature::size() const
        {
            return features_.size();
        }

        std::string tied_lattice_feature::name() const
        {
            return "shared-lattice-feature";
        }

        void tied_lattice_feature::operator()(
            param_t& param,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            if (features_.size() == 0) {
                return;
            }

            if (ebt::in(std::get<0>(e), cache_)) {
                std::vector<real>& v = cache_.at(std::get<0>(e));
                auto& u = param.class_param["[lattice] shared"];
                u.insert(u.end(), v.begin(), v.end());
                return;
            }

            std::vector<real> result;

            auto& feat = fst.fst1->data->features.at(std::get<0>(e));

            for (auto& f: features_) {
                if (ebt::startswith(f, "@")) {
                    result.push_back(feat(f.substr(1)));
                } else {
                    std::cout << "Features " << f << " are not expected to be lexicalized." << std::endl;
                    exit(1);
                }
            }

            cache_[std::get<0>(e)] = result;

            auto& u = param.class_param["[lattice] shared"];
            u.insert(u.end(), result.begin(), result.end());
        }

    }

}
