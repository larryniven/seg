#ifndef PAIR_SCRF_H
#define PAIR_SCRF_H

#include "scrf/experimental/scrf.h"
#include "scrf/experimental/scrf_feat.h"
#include "scrf/experimental/scrf_cost.h"

namespace iscrf {

    namespace second_order {

        struct pair_scrf;

        template <class vector>
        struct pair_scrf_data {

            std::shared_ptr<ilat::pair_fst> fst;
            std::shared_ptr<std::vector<std::tuple<int, int>>> topo_order;
            std::shared_ptr<scrf::scrf_weight<ilat::pair_fst>> weight_func;
            std::shared_ptr<scrf::scrf_feature<ilat::pair_fst, vector>> feature_func;
            std::shared_ptr<scrf::scrf_weight<ilat::pair_fst>> cost_func;
            std::shared_ptr<vector> param;
            std::shared_ptr<std::vector<std::string>> features;

        };

    }

}

namespace scrf {

    template <class vec>
    struct scrf_data_trait<iscrf::second_order::pair_scrf_data<vec>> {
        using base_fst = ilat::pair_fst;
        using path_maker = ilat::pair_fst_path_maker;
        using edge = std::tuple<int, int>;
        using vertex = std::tuple<int, int>;
        using symbol = int;
        using fst = scrf_fst<iscrf::second_order::pair_scrf_data<vec>>;
        using vector = vec;
    };

}

namespace iscrf {

    namespace second_order {

        template <class vector, class lexicalizer>
        scrf::composite_feature<ilat::pair_fst, vector> make_feat(
            scrf::feat_dim_alloc& alloc,
            std::vector<std::string> features,
            std::vector<std::vector<double>> const& frames,
            std::unordered_map<std::string, std::string> const& args);

        namespace sparse {

            struct pair_fst_lexicalizer
                : public scrf::lexicalizer<ilat::pair_fst, scrf::sparse_vec> {

                double* lex(scrf::feat_dim_alloc const& alloc, int order,
                    scrf::sparse_vec& feat, ilat::pair_fst const& fst,
                    ilat::pair_fst::edge e) const override;
            };

            struct backoff_cost
                : public scrf::scrf_weight<ilat::pair_fst> {

                virtual double operator()(ilat::pair_fst const& f,
                    ilat::pair_fst::edge e) const override;

            };

            struct inference_args {
                int min_seg;
                int max_seg;
                scrf::sparse_vec param;
                std::unordered_map<std::string, int> label_id;
                std::vector<std::string> id_label;
                std::vector<int> labels;
                std::vector<std::string> features;
                ilat::fst lm;
                std::unordered_map<std::string, std::string> args;
            };
            
            struct sample {
                std::vector<std::vector<double>> frames;
            
                scrf::feat_dim_alloc graph_alloc;
            
                pair_scrf_data<scrf::sparse_vec> graph_data;
            
                sample(inference_args const& args);
            };
            
            void make_lattice(ilat::fst const& lat, sample& s, inference_args const& i_args);
            
            struct learning_args
                : public inference_args {
            
                scrf::sparse_vec opt_data;
                double step_size;
                double momentum;
                std::vector<int> sils;
            };
            
            struct learning_sample
                : public sample {
            
                std::vector<segcost::segment<int>> gold_segs;
            
                scrf::feat_dim_alloc gold_alloc;
            
                pair_scrf_data<scrf::sparse_vec> gold_data;
            
                learning_sample(learning_args const& args);
            };
            
            learning_args parse_learning_args(
                std::unordered_map<std::string, std::string> const& args);

            void make_min_cost_gold(learning_sample& s, learning_args const& l_args);

        }

        namespace dense {

            struct pair_fst_lexicalizer
                : public scrf::lexicalizer<ilat::pair_fst, scrf::dense_vec> {

                double* lex(scrf::feat_dim_alloc const& alloc, int order,
                    scrf::dense_vec& feat, ilat::pair_fst const& fst,
                    ilat::pair_fst::edge e) const override;
            };

        }

    }

}

namespace iscrf {

    namespace second_order {

        template <class vector, class lexicalizer>
        scrf::composite_feature<ilat::pair_fst, vector> make_feat(
            scrf::feat_dim_alloc& alloc,
            std::vector<std::string> features,
            std::vector<std::vector<double>> const& frames,
            std::unordered_map<std::string, std::string> const& args)
        {
            scrf::composite_feature<ilat::pair_fst, vector> result;

            using feat_func = scrf::segment_feature<ilat::pair_fst, vector, lexicalizer>;

            for (auto& k: features) {
                if (ebt::startswith(k, "frame-avg")) {
                    std::vector<std::string> parts = ebt::split(k, "@");
                    int order = 0;
                    if (parts.size() > 1) {
                        order = std::stoi(parts[1]);
                    }

                    int start_dim = -1;
                    int end_dim = -1;
                    std::tie(start_dim, end_dim) = scrf::get_dim(parts[0]);

                    result.features.push_back(std::make_shared<feat_func>(
                        feat_func(alloc, order,
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
                    std::tie(start_dim, end_dim) = scrf::get_dim(parts[0]);

                    result.features.push_back(std::make_shared<feat_func>(
                        feat_func(alloc, order,
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
                    std::tie(start_dim, end_dim) = scrf::get_dim(parts[0]);

                    result.features.push_back(std::make_shared<feat_func>(
                        feat_func(alloc, order,
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
                    std::tie(start_dim, end_dim) = scrf::get_dim(parts[0]);

                    result.features.push_back(std::make_shared<feat_func>(
                        feat_func(alloc, order,
                        std::make_shared<segfeat::right_boundary>(
                            segfeat::right_boundary { start_dim, end_dim }),
                        frames)));
                } else if (ebt::startswith(k, "length-indicator")) {
                    std::vector<std::string> parts = ebt::split(k, "@");
                    int order = 0;
                    if (parts.size() > 1) {
                        order = std::stoi(parts[1]);
                    }

                    if (!ebt::in(std::string("max-seg"), args)) {
                        std::cerr << "--max-seg is required" << std::endl;
                        exit(1);
                    }

                    int max_seg = std::stoi(args.at("max-seg"));

                    result.features.push_back(std::make_shared<feat_func>(
                        feat_func(alloc, order,
                        std::make_shared<segfeat::length_indicator>(
                            segfeat::length_indicator { max_seg }),
                        frames)));
                } else if (ebt::startswith(k, "bias")) {
                    std::vector<std::string> parts = ebt::split(k, "@");
                    int order = 0;
                    if (parts.size() > 1) {
                        order = std::stoi(parts[1]);
                    }

                    result.features.push_back(std::make_shared<feat_func>(
                        feat_func(alloc, order,
                        std::make_shared<segfeat::bias>(
                            segfeat::bias {}),
                        frames)));
                } else {
                    std::cerr << "unknown feature " << k << std::endl;
                    exit(1);
                }
            }

            return result;
        }

    }

}

#endif
