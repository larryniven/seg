#ifndef PAIR_SCRF_H
#define PAIR_SCRF_H

#include "scrf/experimental/scrf.h"
#include "scrf/experimental/scrf_feat.h"
#include "scrf/experimental/scrf_cost.h"

namespace scrf {

    namespace second_order {

        template <class vector>
        struct pair_scrf
            : public scrf<std::tuple<int, int>, std::tuple<int, int>, int, vector> {

            using vertex = std::tuple<int, int>;
            using edge = std::tuple<int, int>;
            using symbol = int;

            std::shared_ptr<ilat::pair_fst> fst;
            std::vector<vertex> topo_order_cache;

            std::shared_ptr<scrf_weight<ilat::pair_fst>> weight_func;
            std::shared_ptr<scrf_feature<ilat::pair_fst, vector>> feature_func;
            std::shared_ptr<scrf_weight<ilat::pair_fst>> cost_func;

            virtual std::vector<vertex> const& vertices() const override;
            virtual std::vector<edge> const& edges() const override;
            virtual vertex head(edge e) const override;
            virtual vertex tail(edge e) const override;
            virtual std::vector<edge> const& in_edges(vertex v) const override;
            virtual std::vector<edge> const& out_edges(vertex v) const override;
            virtual double weight(edge e) const override;
            virtual int const& input(edge e) const override;
            virtual int const& output(edge e) const override;
            virtual std::vector<vertex> const& initials() const override;
            virtual std::vector<vertex> const& finals() const override;

            virtual long time(vertex v) const override;

            virtual void feature(vector& f, edge e) const override;
            virtual double cost(edge e) const override;

            virtual std::vector<vertex> const& topo_order() const override;

        };

        template <class vector>
        struct pair_scrf_path_maker
            : public fst::path_maker<pair_scrf<vector>> {

            virtual std::shared_ptr<pair_scrf<vector>> operator()(
                std::vector<typename pair_scrf<vector>::edge> const& edges,
                pair_scrf<vector> const& f) const override;
        };

        template <class vector, class lexicalizer>
        composite_feature<ilat::pair_fst, vector> make_feat(
            feat_dim_alloc& alloc,
            std::vector<std::string> features,
            std::vector<std::vector<double>> const& frames,
            std::unordered_map<std::string, std::string> const& args);

        namespace sparse {

            struct pair_fst_lexicalizer
                : public lexicalizer<ilat::pair_fst, sparse_vec> {

                double* lex(feat_dim_alloc const& alloc, int order,
                    sparse_vec& feat, ilat::pair_fst const& fst,
                    ilat::pair_fst::edge e) const override;
            };

            struct backoff_cost
                : public scrf_weight<ilat::pair_fst> {

                virtual double operator()(ilat::pair_fst const& f,
                    ilat::pair_fst::edge e) const override;

            };

            struct inference_args {
                int min_seg;
                int max_seg;
                sparse_vec param;
                std::unordered_map<std::string, int> label_id;
                std::vector<std::string> id_label;
                std::vector<int> labels;
                std::vector<std::string> features;
                ilat::fst lm;
                std::unordered_map<std::string, std::string> args;
            };
            
            struct sample {
                std::vector<std::vector<double>> frames;
            
                feat_dim_alloc graph_alloc;
            
                pair_scrf<sparse_vec> graph;
                std::shared_ptr<pair_scrf<sparse_vec>> graph_path;
            
                sample(inference_args const& args);
            };
            
            void make_lattice(ilat::fst const& lat, sample& s, inference_args const& i_args);
            
            struct learning_args
                : public inference_args {
            
                sparse_vec opt_data;
                double step_size;
                double momentum;
                std::vector<int> sils;
            };
            
            struct learning_sample
                : public sample {
            
                std::vector<segcost::segment<int>> gold_segs;
            
                feat_dim_alloc gold_alloc;
            
                std::shared_ptr<pair_scrf<sparse_vec>> gold;
            
                std::shared_ptr<seg_cost<ilat::pair_fst>> cost;
            
                learning_sample(learning_args const& args);
            };
            
            learning_args parse_learning_args(
                std::unordered_map<std::string, std::string> const& args);

            void make_min_cost_gold(learning_sample& s, learning_args const& l_args);

        }

        namespace dense {

            struct pair_fst_lexicalizer
                : public lexicalizer<ilat::pair_fst, dense_vec> {

                double* lex(feat_dim_alloc const& alloc, int order,
                    dense_vec& feat, ilat::pair_fst const& fst,
                    ilat::pair_fst::edge e) const override;
            };

        }

    }

}

namespace scrf {

    namespace second_order {

        template <class vector>
        std::vector<typename pair_scrf<vector>::vertex> const&
        pair_scrf<vector>::vertices() const
        {
            return fst->vertices();
        }

        template <class vector>
        std::vector<typename pair_scrf<vector>::edge> const&
        pair_scrf<vector>::edges() const
        {
            return fst->edges();
        }

        template <class vector>
        typename pair_scrf<vector>::vertex
        pair_scrf<vector>::head(typename pair_scrf<vector>::edge e) const
        {
            return fst->head(e);
        }

        template <class vector>
        typename pair_scrf<vector>::vertex
        pair_scrf<vector>::tail(typename pair_scrf<vector>::edge e) const
        {
            return fst->tail(e);
        }

        template <class vector>
        std::vector<typename pair_scrf<vector>::edge> const&
        pair_scrf<vector>::in_edges(typename pair_scrf<vector>::vertex v) const
        {
            return fst->in_edges(v);
        }

        template <class vector>
        std::vector<typename pair_scrf<vector>::edge> const&
        pair_scrf<vector>::out_edges(typename pair_scrf<vector>::vertex v) const
        {
            return fst->out_edges(v);
        }

        template <class vector>
        double pair_scrf<vector>::weight(typename pair_scrf<vector>::edge e) const
        {
            return (*weight_func)(*fst, e);
        }

        template <class vector>
        int const& pair_scrf<vector>::input(typename pair_scrf<vector>::edge e) const
        {
            return fst->input(e);
        }

        template <class vector>
        int const& pair_scrf<vector>::output(typename pair_scrf<vector>::edge e) const
        {
            return fst->output(e);
        }

        template <class vector>
        std::vector<typename pair_scrf<vector>::vertex> const&
        pair_scrf<vector>::initials() const
        {
            return fst->initials();
        }

        template <class vector>
        std::vector<typename pair_scrf<vector>::vertex> const&
        pair_scrf<vector>::finals() const
        {
            return fst->finals();
        }

        template <class vector>
        long pair_scrf<vector>::time(typename pair_scrf<vector>::edge e) const
        {
            return fst->time(e);
        }

        template <class vector>
        void pair_scrf<vector>::feature(vector& f, typename pair_scrf::edge e) const
        {
            (*feature_func)(f, *fst, e);
        }

        template <class vector>
        double pair_scrf<vector>::cost(typename pair_scrf<vector>::edge e) const
        {
            return (*cost_func)(*fst, e);
        }

        template <class vector>
        std::vector<typename pair_scrf<vector>::vertex> const&
        pair_scrf<vector>::topo_order() const
        {
            return topo_order_cache;
        }

        template <class vector>
        std::shared_ptr<pair_scrf<vector>> pair_scrf_path_maker<vector>::operator()(
            std::vector<typename pair_scrf<vector>::edge> const& edges,
            pair_scrf<vector> const& f) const
        {
            pair_scrf<vector> result;

            result.fst = ilat::pair_fst_path_maker()(edges, *f.fst);
            result.topo_order_cache = fst::topo_order(*result.fst);
            result.weight_func = f.weight_func;
            result.feature_func = f.feature_func;
            result.cost_func = f.cost_func;

            return std::make_shared<pair_scrf<vector>>(result);
        }

        template <class vector, class lexicalizer>
        composite_feature<ilat::pair_fst, vector> make_feat(
            feat_dim_alloc& alloc,
            std::vector<std::string> features,
            std::vector<std::vector<double>> const& frames,
            std::unordered_map<std::string, std::string> const& args)
        {
            composite_feature<ilat::pair_fst, vector> result;

            using feat_func = segment_feature<ilat::pair_fst, vector, lexicalizer>;

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
                    std::tie(start_dim, end_dim) = get_dim(parts[0]);

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
                    std::tie(start_dim, end_dim) = get_dim(parts[0]);

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
                    std::tie(start_dim, end_dim) = get_dim(parts[0]);

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
