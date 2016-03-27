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
        std::vector<std::vector<real>> const& frames,
        std::unordered_map<std::string, std::string> const& args)
    {
        composite_feature result;

        for (auto& k: features) {
            if (ebt::startswith(k, "frame-avg")) {
                std::vector<std::string> parts = ebt::split(k, "@");
                int order = 0;
                if (parts.size() > 1) {
                    order = std::stoi(parts[1]);
                }

                int start_dim = -1;
                int end_dim = -1;
                std::tie(start_dim, end_dim) = get_dim(parts[0]);

                result.features.push_back(std::make_shared<segment_feature>(
                    segment_feature(order,
                    std::make_shared<segfeat::frame_avg>(
                        segfeat::frame_avg { frames, start_dim, end_dim }),
                    frames)));
            } else if (ebt::startswith(k, "frame-samples")) {
                std::vector<std::string> parts = ebt::split(k, "@");
                int order = 0;
                if (parts.size() > 1) {
                    order = std::stoi(parts[1]);
                }

                int start_dim = -1;
                int end_dim = -1;
                std::tie(start_dim, end_dim) = get_dim(parts[0]);

                result.features.push_back(std::make_shared<segment_feature>(
                    segment_feature(order,
                    std::make_shared<segfeat::frame_samples>(
                        segfeat::frame_samples { 3, start_dim, end_dim }),
                    frames)));
            } else if (ebt::startswith(k, "left-boundary")) {
                std::vector<std::string> parts = ebt::split(k, "@");
                int order = 0;
                if (parts.size() > 1) {
                    order = std::stoi(parts[1]);
                }

                int start_dim = -1;
                int end_dim = -1;
                std::tie(start_dim, end_dim) = get_dim(parts[0]);

                result.features.push_back(std::make_shared<segment_feature>(
                    segment_feature(order,
                    std::make_shared<segfeat::left_boundary>(
                        segfeat::left_boundary { start_dim, end_dim }),
                    frames)));
            } else if (ebt::startswith(k, "right-boundary")) {
                std::vector<std::string> parts = ebt::split(k, "@");
                int order = 0;
                if (parts.size() > 1) {
                    order = std::stoi(parts[1]);
                }

                int start_dim = -1;
                int end_dim = -1;
                std::tie(start_dim, end_dim) = get_dim(parts[0]);

                result.features.push_back(std::make_shared<segment_feature>(
                    segment_feature(order,
                    std::make_shared<segfeat::right_boundary>(
                        segfeat::right_boundary { start_dim, end_dim }),
                    frames)));
            } else if (ebt::startswith(k, "length-indicator")) {
                std::vector<std::string> parts = ebt::split(k, "@");
                int order = 0;
                if (parts.size() > 1) {
                    order = std::stoi(parts[1]);
                }

                int max_seg = std::stoi(args.at("max-seg"));

                result.features.push_back(std::make_shared<segment_feature>(
                    segment_feature(order,
                    std::make_shared<segfeat::length_indicator>(
                        segfeat::length_indicator { max_seg }),
                    frames)));
            } else if (ebt::startswith(k, "frame")) {
                result.features.push_back(std::make_shared<feature::frame_feature>(
                    feature::frame_feature { frames, args }));
            } else if (ebt::startswith(k, "lm-score")) {
                result.features.push_back(std::make_shared<feature::lm_score>(
                    feature::lm_score{}));
            } else if (ebt::startswith(k, "lattice-score")) {
                result.features.push_back(std::make_shared<feature::lattice_score>(
                    feature::lattice_score{}));
            } else if (ebt::startswith(k, "ext")) {
                std::vector<std::string> parts = ebt::split(k, "@");
                int order = 0;
                if (parts.size() > 1) {
                    order = std::stoi(parts[1]);
                }

                parts = ebt::split(parts[0], ":");
                parts = ebt::split(parts[1], "+");
                std::vector<int> dims;

                for (auto& p: parts) {
                    std::vector<std::string> range = ebt::split(p, "-");
                    if (range.size() == 2) {
                        for (int i = std::stoi(range[0]); i <= std::stoi(range[1]); ++i) {
                            dims.push_back(i);
                        }
                    } else if (range.size() == 1) {
                        dims.push_back(std::stoi(p));
                    } else {
                        std::cerr << "unknown external feature format: " << k << std::endl;
                    }
                }

                result.features.push_back(std::make_shared<feature::external_feature>(
                    feature::external_feature { order, dims }));
            } else if (ebt::startswith(k, "bias")) {
                std::vector<std::string> parts = ebt::split(k, "@");
                int order = 0;
                if (parts.size() > 1) {
                    order = std::stoi(parts[1]);
                }

                result.features.push_back(std::make_shared<segment_feature>(
                    segment_feature(order,
                    std::make_shared<segfeat::bias>(
                        segfeat::bias {}),
                    frames)));
            } else if (ebt::startswith(k, "quad-length")) {
                std::vector<std::string> parts = ebt::split(k, "@");
                int order = 0;
                if (parts.size() > 1) {
                    order = std::stoi(parts[1]);
                }

                result.features.push_back(std::make_shared<feature::quad_length>(
                    feature::quad_length { order, args }));
            } else {
                std::cerr << "unknown feature " << k << std::endl;
                exit(1);
            }
        }

        return result;
    }

    namespace first_order {

        composite_feature make_feat(
            feat_dim_alloc& alloc,
            std::vector<std::string> features,
            std::vector<std::vector<real>> const& frames,
            std::unordered_map<std::string, std::string> const& args)
        {
            composite_feature result;

            for (auto& k: features) {
                if (ebt::startswith(k, "frame-avg")) {
                    std::vector<std::string> parts = ebt::split(k, "@");
                    int order = 0;
                    if (parts.size() > 1) {
                        order = std::stoi(parts[1]);
                    }

                    int start_dim = -1;
                    int end_dim = -1;
                    std::tie(start_dim, end_dim) = get_dim(parts[0]);

                    result.features.push_back(std::make_shared<segment_feature>(
                        segment_feature(alloc, order,
                        std::make_shared<segfeat::la::frame_avg>(
                            segfeat::la::frame_avg { frames, start_dim, end_dim }),
                        frames)));
                } else if (ebt::startswith(k, "frame-samples")) {
                    std::vector<std::string> parts = ebt::split(k, "@");
                    int order = 0;
                    if (parts.size() > 1) {
                        order = std::stoi(parts[1]);
                    }

                    int start_dim = -1;
                    int end_dim = -1;
                    std::tie(start_dim, end_dim) = get_dim(parts[0]);

                    result.features.push_back(std::make_shared<segment_feature>(
                        segment_feature(alloc, order,
                        std::make_shared<segfeat::la::frame_samples>(
                            segfeat::la::frame_samples { 3, start_dim, end_dim }),
                        frames)));
                } else if (ebt::startswith(k, "left-boundary")) {
                    std::vector<std::string> parts = ebt::split(k, "@");
                    int order = 0;
                    if (parts.size() > 1) {
                        order = std::stoi(parts[1]);
                    }

                    int start_dim = -1;
                    int end_dim = -1;
                    std::tie(start_dim, end_dim) = get_dim(parts[0]);

                    result.features.push_back(std::make_shared<segment_feature>(
                        segment_feature(alloc, order,
                        std::make_shared<segfeat::la::left_boundary>(
                            segfeat::la::left_boundary { start_dim, end_dim }),
                        frames)));
                } else if (ebt::startswith(k, "right-boundary")) {
                    std::vector<std::string> parts = ebt::split(k, "@");
                    int order = 0;
                    if (parts.size() > 1) {
                        order = std::stoi(parts[1]);
                    }

                    int start_dim = -1;
                    int end_dim = -1;
                    std::tie(start_dim, end_dim) = get_dim(parts[0]);

                    result.features.push_back(std::make_shared<segment_feature>(
                        segment_feature(alloc, order,
                        std::make_shared<segfeat::la::right_boundary>(
                            segfeat::la::right_boundary { start_dim, end_dim }),
                        frames)));
                } else if (ebt::startswith(k, "length-indicator")) {
                    std::vector<std::string> parts = ebt::split(k, "@");
                    int order = 0;
                    if (parts.size() > 1) {
                        order = std::stoi(parts[1]);
                    }

                    int max_seg = std::stoi(args.at("max-seg"));

                    result.features.push_back(std::make_shared<segment_feature>(
                        segment_feature(alloc, order,
                        std::make_shared<segfeat::la::length_indicator>(
                            segfeat::la::length_indicator { max_seg }),
                        frames)));
                } else if (ebt::startswith(k, "frame")) {
                    result.features.push_back(std::make_shared<feature::frame_feature>(
                        feature::frame_feature { alloc, frames, args }));
                } else if (ebt::startswith(k, "lattice-score")) {
                    result.features.push_back(std::make_shared<feature::lattice_score>(
                        feature::lattice_score { alloc }));
                } else if (ebt::startswith(k, "ext")) {
                    std::vector<std::string> parts = ebt::split(k, "@");
                    int order = 0;
                    if (parts.size() > 1) {
                        order = std::stoi(parts[1]);
                    }

                    parts = ebt::split(parts[0], ":");
                    parts = ebt::split(parts[1], "+");
                    std::vector<int> dims;

                    for (auto& p: parts) {
                        std::vector<std::string> range = ebt::split(p, "-");
                        if (range.size() == 2) {
                            for (int i = std::stoi(range[0]); i <= std::stoi(range[1]); ++i) {
                                dims.push_back(i);
                            }
                        } else if (range.size() == 1) {
                            dims.push_back(std::stoi(p));
                        } else {
                            std::cerr << "unknown external feature format: " << k << std::endl;
                        }
                    }

                    result.features.push_back(std::make_shared<feature::external_feature>(
                        feature::external_feature { alloc, order, dims }));
                } else if (ebt::startswith(k, "bias")) {
                    std::vector<std::string> parts = ebt::split(k, "@");
                    int order = 0;
                    if (parts.size() > 1) {
                        order = std::stoi(parts[1]);
                    }

                    result.features.push_back(std::make_shared<segment_feature>(
                        segment_feature(alloc, order,
                        std::make_shared<segfeat::la::bias>(
                            segfeat::la::bias {}),
                        frames)));
                } else if (ebt::startswith(k, "quad-length")) {
                    std::vector<std::string> parts = ebt::split(k, "@");
                    int order = 0;
                    if (parts.size() > 1) {
                        order = std::stoi(parts[1]);
                    }

                    result.features.push_back(std::make_shared<feature::quad_length>(
                        feature::quad_length { alloc, order, args }));
                } else if (ebt::startswith(k, "max-hits")) {
                    std::vector<std::string> parts = ebt::split(k, "@");
                    int order = 0;
                    if (parts.size() > 1) {
                        order = std::stoi(parts[1]);
                    }

                    int max_seg = std::stoi(args.at("max-seg"));

                    result.features.push_back(std::make_shared<feature::max_hits>(
                        feature::max_hits { alloc, order, 0.9, max_seg, frames }));
                } else {
                    std::cerr << "unknown feature " << k << std::endl;
                    exit(1);
                }
            }

            return result;
        }

    }

}
