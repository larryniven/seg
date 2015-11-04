#include "scrf/scrf_feat.h"
#include "scrf/nn.h"
#include "scrf/weiran.h"
#include <cassert>

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

    }

    lexicalized_feature::lexicalized_feature(
        int order,
        std::shared_ptr<segfeat::feature> feat_func,
        std::vector<std::vector<real>> const& frames)
        : order(order), feat_func(feat_func), frames(frames)
    {}

    int lexicalized_feature::size() const
    {
        return 0;
    }

    std::string lexicalized_feature::name() const
    {
        return "";
    }

    void lexicalized_feature::operator()(
        param_t& feat,
        fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        segfeat::feat_t raw_feat;

        lattice::fst const& lat = *fst.fst1;
        int tail_time = lat.data->vertices.at(std::get<0>(fst.tail(e))).time;
        int head_time = lat.data->vertices.at(std::get<0>(fst.head(e))).time;

        (*feat_func)(raw_feat, frames, tail_time, head_time);

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

        feat.class_param[label_tuple] = std::move(raw_feat);
    }

#if 0
    composite_feature make_feature(
        std::vector<std::string> features,
        std::vector<std::vector<real>> const& inputs, int max_seg)
    {
        return make_feature(features, inputs, max_seg,
            std::vector<real>(), std::vector<real>(), nn::nn_t());
    }

    composite_feature make_feature(
        std::vector<std::string> features,
        std::vector<std::vector<real>> const& inputs, int max_seg,
        std::vector<real> const& cm_mean,
        std::vector<real> const& cm_stddev,
        nn::nn_t const& nn)
    {
        composite_feature result { "all" };
    
        composite_feature lex_lattice_feat { "lex-lattice-feat" };
        composite_feature tied_lattice_feat { "tied-lattice-feat" };
        composite_feature lm_feat { "lm-feat" };
        composite_feature label_feat { "label-feat" };
        composite_feature rest_feat { "rest-feat" };

        for (auto& v: features) {
            if (ebt::startswith(v, "frame-avg")) {
                std::vector<std::string> parts = ebt::split(v, ":");
                if (parts.size() == 2) {
                    std::vector<std::string> indices = ebt::split(parts.back(), "-");
                    lex_lattice_feat.features.push_back(std::make_shared<feature::frame_avg>(
                        feature::frame_avg { inputs, std::stoi(indices.at(0)), std::stoi(indices.at(1)) }));
                } else {
                    lex_lattice_feat.features.push_back(std::make_shared<feature::frame_avg>(
                        feature::frame_avg { inputs }));
                }
            } else if (ebt::startswith(v, "frame-samples")) {
                std::vector<std::string> parts = ebt::split(v, ":");
                if (parts.size() == 2) {
                    std::vector<std::string> indices = ebt::split(parts.back(), "-");
                    lex_lattice_feat.features.push_back(std::make_shared<feature::frame_samples>(
                        feature::frame_samples { inputs, 3, std::stoi(indices.at(0)), std::stoi(indices.at(1)) }));
                } else {
                    lex_lattice_feat.features.push_back(std::make_shared<feature::frame_samples>(
                        feature::frame_samples { inputs, 3 }));
                }
            } else if (ebt::startswith(v, "left-boundary")) {
                std::vector<std::string> parts = ebt::split(v, ":");
                if (parts.size() == 2) {
                    std::vector<std::string> indices = ebt::split(parts.back(), "-");
                    rest_feat.features.push_back(std::make_shared<feature::left_boundary>(
                        feature::left_boundary { inputs, std::stoi(indices.at(0)), std::stoi(indices.at(1)) }));
                } else {
                    rest_feat.features.push_back(std::make_shared<feature::left_boundary>(
                        feature::left_boundary { inputs }));
                }
            } else if (ebt::startswith(v, "right-boundary")) {
                std::vector<std::string> parts = ebt::split(v, ":");
                if (parts.size() == 2) {
                    std::vector<std::string> indices = ebt::split(parts.back(), "-");
                    rest_feat.features.push_back(std::make_shared<feature::right_boundary>(
                        feature::right_boundary { inputs, std::stoi(indices.at(0)),
                        std::stoi(indices.at(1)) }));
                } else {
                    rest_feat.features.push_back(std::make_shared<feature::right_boundary>(
                        feature::right_boundary { inputs }));
                }
            } else if (ebt::startswith(v, "length-indicator")) {
                lex_lattice_feat.features.push_back(std::make_shared<feature::length_indicator>(
                    feature::length_indicator { max_seg }));
            } else if (ebt::startswith(v, "length-value")) {
                lex_lattice_feat.features.push_back(std::make_shared<feature::length_value>(
                    feature::length_value { max_seg }));
            } else if (ebt::startswith(v, "bias")) {
                label_feat.features.push_back(std::make_shared<feature::bias>(feature::bias{}));
            } else if (ebt::startswith(v, "lm-score")) {
                lm_feat.features.push_back(std::make_shared<feature::lm_score>(feature::lm_score{}));
            } else if (ebt::startswith(v, "lattice-score")) {
                tied_lattice_feat.features.push_back(std::make_shared<feature::lattice_score>(
                    feature::lattice_score{}));
            } else if (ebt::startswith(v, "nn")) {
                std::vector<std::string> parts = ebt::split(v, ":");
                if (parts.size() == 2) {
                    std::vector<std::string> indices = ebt::split(parts.back(), "-");
                    lex_lattice_feat.features.push_back(std::make_shared<nn::nn_feature>(
                        nn::nn_feature { inputs, cm_mean, cm_stddev, nn,
                        std::stoi(indices.at(0)), std::stoi(indices.at(1)) }));
                } else {
                    lex_lattice_feat.features.push_back(std::make_shared<nn::nn_feature>(
                        nn::nn_feature { inputs, cm_mean, cm_stddev, nn }));
                }
            } else if (ebt::startswith(v, "weiran")) {
                std::vector<std::string> parts = ebt::split(v, ":");
                if (parts.size() == 2) {
                    std::vector<std::string> indices = ebt::split(parts.back(), "-");
                    lex_lattice_feat.features.push_back(std::make_shared<weiran::weiran_feature>(
                        weiran::weiran_feature { inputs, cm_mean, cm_stddev, nn,
                        std::stoi(indices.at(0)), std::stoi(indices.at(1)) }));
                } else {
                    lex_lattice_feat.features.push_back(std::make_shared<weiran::weiran_feature>(
                        weiran::weiran_feature { inputs, cm_mean, cm_stddev, nn }));
                }
            } else {
                std::cout << "unknown featre type " << v << std::endl;
                exit(1);
            }
        }
    
        result.features.push_back(std::make_shared<composite_feature>(lex_lattice_feat));
        result.features.push_back(std::make_shared<composite_feature>(tied_lattice_feat));
        result.features.push_back(std::make_shared<composite_feature>(lm_feat));
        result.features.push_back(std::make_shared<composite_feature>(label_feat));
        result.features.push_back(std::make_shared<composite_feature>(rest_feat));

        return result;
    }
#endif

    std::pair<int, int> get_dim(std::string feat)
    {
        std::vector<std::string> parts = ebt::split(feat, ":");
        int start_dim = -1;
        int end_dim = -1;
        if (parts.size() == 2) {
            std::vector<std::string> indices = ebt::split(parts.back(), "-");
            start_dim = std::stoi(indices.at(0));
            end_dim = std::stoi(indices.at(1));
        }

        return std::make_pair(start_dim, end_dim);
    }

    composite_feature make_feature2(
        std::vector<std::string> features,
        std::vector<std::vector<real>> const& frames)
    {
        composite_feature result { "all" };
    
        composite_feature lex_lattice_feat { "lex-lattice-feat" };
        composite_feature tied_lattice_feat { "tied-lattice-feat" };
        composite_feature lm_feat { "lm-feat" };
        composite_feature label_feat { "label-feat" };
        composite_feature rest_feat { "rest-feat" };

        std::vector<std::string> other_feat;
        std::vector<std::vector<std::string>> seg_feature_order;
        seg_feature_order.resize(2);

        for (auto& v: features) {
            std::vector<std::string> parts = ebt::split(v, "@");

            if (parts.size() == 2) {
                if (parts[1] == "0") {
                    seg_feature_order[0].push_back(parts[0]);
                } else if (parts[1] == "1") {
                    seg_feature_order[1].push_back(parts[0]);
                } else {
                    std::cerr << "order " << parts[1] << " not implemented" << std::endl;
                    exit(1);
                }
            } else {
                other_feat.push_back(v);
            }
        }

        for (int i = 0; i < seg_feature_order.size(); ++i) {
            segfeat::composite_feature seg_feat;

            if (seg_feature_order.at(i).size() == 0) {
                continue;
            }

            for (auto& v: seg_feature_order.at(i)) {

                std::shared_ptr<segfeat::feat_t> adapter;

                if (ebt::startswith(v, "frame-avg")) {
                    int start_dim = -1;
                    int end_dim = -1;
                    std::tie(start_dim, end_dim) = get_dim(v);

                    seg_feat.features.push_back(std::make_shared<segfeat::frame_avg>(
                        segfeat::frame_avg { start_dim, end_dim }));
                } else if (ebt::startswith(v, "frame-samples")) {
                    int start_dim = -1;
                    int end_dim = -1;
                    std::tie(start_dim, end_dim) = get_dim(v);

                    seg_feat.features.push_back(std::make_shared<segfeat::frame_samples>(
                        segfeat::frame_samples { 3, start_dim, end_dim }));
                } else if (ebt::startswith(v, "left-boundary")) {
                    int start_dim = -1;
                    int end_dim = -1;
                    std::tie(start_dim, end_dim) = get_dim(v);

                    seg_feat.features.push_back(std::make_shared<segfeat::left_boundary>(
                        segfeat::left_boundary { start_dim, end_dim }));
                } else if (ebt::startswith(v, "right-boundary")) {
                    int start_dim = -1;
                    int end_dim = -1;
                    std::tie(start_dim, end_dim) = get_dim(v);

                    seg_feat.features.push_back(std::make_shared<segfeat::right_boundary>(
                        segfeat::right_boundary { start_dim, end_dim }));
                } else if (ebt::startswith(v, "length-indicator")) {
                    seg_feat.features.push_back(std::make_shared<segfeat::length_indicator>(
                        segfeat::length_indicator { 30 }));
                } else if (ebt::startswith(v, "bias")) {
                    seg_feat.features.push_back(std::make_shared<segfeat::bias>(
                        segfeat::bias {}));
                } else {
                    std::cerr << "unknown feature " << v << std::endl;
                    exit(1);
                }
            }

            scrf::lexicalized_feature lex_feat { i,
                std::make_shared<segfeat::composite_feature>(seg_feat), frames };

            lex_lattice_feat.features.push_back(std::make_shared<scrf::lexicalized_feature>(lex_feat));
        }

        for (auto& v: other_feat) {
            if (v == "lm-score") {
                lm_feat.features.push_back(std::make_shared<feature::lm_score>(feature::lm_score{}));
            } else if (v == "lattice-score") {
                tied_lattice_feat.features.push_back(std::make_shared<feature::lattice_score>(
                    feature::lattice_score{}));
            } else {
                std::cerr << "unknown feature " << v << std::endl;
                exit(1);
            }
        }

        result.features.push_back(std::make_shared<composite_feature>(lex_lattice_feat));
        result.features.push_back(std::make_shared<composite_feature>(tied_lattice_feat));
        result.features.push_back(std::make_shared<composite_feature>(lm_feat));
        result.features.push_back(std::make_shared<composite_feature>(label_feat));
        result.features.push_back(std::make_shared<composite_feature>(rest_feat));

        return result;
    }

}
