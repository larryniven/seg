#include "scrf/make_feat.h"

namespace scrf {

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

    composite_feature make_feat(
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
        seg_feature_order.resize(3);

        for (auto& v: features) {
            std::vector<std::string> parts = ebt::split(v, "@");

            if (parts.size() == 2) {
                if (parts[1] == "0") {
                    seg_feature_order[0].push_back(parts[0]);
                } else if (parts[1] == "1") {
                    seg_feature_order[1].push_back(parts[0]);
                } else if (parts[1] == "2") {
                    seg_feature_order[2].push_back(parts[0]);
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

            scrf::feature::external_feature ext_feat;
            ext_feat.order = i;

            if (seg_feature_order.at(i).size() == 0) {
                continue;
            }

            for (auto& v: seg_feature_order.at(i)) {

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
                } else if (ebt::startswith(v, "ext")) {
                    std::vector<std::string> parts = ebt::split(v, ":");
                    ext_feat.feature_keys.insert(parts[1]);
                } else {
                    std::cerr << "unknown feature " << v << std::endl;
                    exit(1);
                }
            }

            if (seg_feat.features.size() > 0) {
                scrf::lexicalized_feature lex_feat { i,
                    std::make_shared<segfeat::composite_feature>(seg_feat), frames };

                if (i == 1) {
                    lex_lattice_feat.features.push_back(std::make_shared<scrf::lexicalized_feature>(lex_feat));
                } else if (i == 0 || i == 2) {
                    rest_feat.features.push_back(std::make_shared<scrf::lexicalized_feature>(lex_feat));
                } else {
                    std::cerr << "unknown feature order " << i << std::endl;
                    exit(1);
                }
            }

            if (ext_feat.feature_keys.size() > 0) {
                lex_lattice_feat.features.push_back(std::make_shared<scrf::feature::external_feature>(ext_feat));
            }
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
