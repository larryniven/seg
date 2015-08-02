#include "scrf/make_feature.h"

namespace scrf {

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

        std::vector<std::string> lex_lat_features;
        std::vector<std::string> tied_lat_features;

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
            } else if (ebt::startswith(v, "lex@")) {
                lex_lat_features.push_back(v);
            } else if (ebt::startswith(v, "@")) {
                tied_lat_features.push_back(v);
            } else {
                std::cout << "unknown featre type " << v << std::endl;
                exit(1);
            }
        }
    
        lex_lattice_feat.features.push_back(std::make_shared<feature::lex_lattice_feature>(
            feature::lex_lattice_feature { lex_lat_features }));
        tied_lattice_feat.features.push_back(std::make_shared<feature::tied_lattice_feature>(
            feature::tied_lattice_feature { tied_lat_features }));

        result.features.push_back(std::make_shared<composite_feature>(lex_lattice_feat));
        result.features.push_back(std::make_shared<composite_feature>(tied_lattice_feat));
        result.features.push_back(std::make_shared<composite_feature>(lm_feat));
        result.features.push_back(std::make_shared<composite_feature>(label_feat));
        result.features.push_back(std::make_shared<composite_feature>(rest_feat));

        return result;
    }

    composite_weight make_weight(
        param_t const& param,
        composite_feature const& feat)
    {
        composite_weight result;

        composite_feature lattice_feat { "lattice-feat" };
        lattice_feat.features.push_back(feat.features[0]);
        lattice_feat.features.push_back(feat.features[1]);

        score::lattice_score lattice_score { param, std::make_shared<composite_feature>(lattice_feat) };
        score::lm_score lm_score { param, feat.features[2] };
        score::label_score label_score { param, feat.features[3] };
        score::linear_score rest_score { param, feat.features[4] };

        result.weights.push_back(std::make_shared<score::lattice_score>(lattice_score));
        result.weights.push_back(std::make_shared<score::lm_score>(lm_score));
        result.weights.push_back(std::make_shared<score::label_score>(label_score));
        result.weights.push_back(std::make_shared<score::linear_score>(rest_score));

        return result;
    }

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
                seg_feature_order[0].push_back(parts[0]);
            }
        }

        for (int i = 0; i < seg_feature_order.size(); ++i) {
            segfeat::composite_feature seg_feat;

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

        result.features.push_back(std::make_shared<composite_feature>(lex_lattice_feat));
        result.features.push_back(std::make_shared<composite_feature>(tied_lattice_feat));
        result.features.push_back(std::make_shared<composite_feature>(lm_feat));
        result.features.push_back(std::make_shared<composite_feature>(label_feat));
        result.features.push_back(std::make_shared<composite_feature>(rest_feat));

        return result;
    }

}
