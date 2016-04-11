#ifndef PAIR_SCRF_H
#define PAIR_SCRF_H

#include "scrf/scrf.h"
#include "scrf/scrf_feat.h"
#include "scrf/scrf_cost.h"
#include "scrf/scrf_util.h"

namespace scrf {

    namespace experimental {

        struct pair_scrf
            : public scrf<std::tuple<int, int>, std::tuple<int, int>, int, sparse_vec> {

            using vertex = std::tuple<int, int>;
            using edge = std::tuple<int, int>;
            using symbol = int;

            std::shared_ptr<ilat::pair_fst> fst;
            std::vector<vertex> topo_order_cache;

            std::shared_ptr<scrf_weight<ilat::pair_fst>> weight_func;
            std::shared_ptr<scrf_feature<ilat::pair_fst, sparse_vec>> feature_func;
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

            virtual void feature(sparse_vec& f, edge e) const override;
            virtual double cost(edge e) const override;

            virtual std::vector<vertex> const& topo_order() const override;

        };

        struct pair_scrf_path_maker
            : public fst::experimental::path_maker<pair_scrf> {

            virtual std::shared_ptr<pair_scrf> operator()(std::vector<typename pair_scrf::edge> const& edges,
                pair_scrf const& f) const override;
        };

        struct pair_fst_lexicalizer
            : public lexicalizer<ilat::pair_fst, sparse_vec> {

            virtual double* operator()(feat_dim_alloc const& alloc,
                int order, sparse_vec& f, ilat::pair_fst const& a,
                ilat::pair_fst::edge e) const override;
        };

        struct backoff_cost
            : public scrf_weight<ilat::pair_fst> {

            virtual double operator()(ilat::pair_fst const& f,
                ilat::pair_fst::edge e) const override;

        };

        std::pair<int, int> get_dim(std::string feat);

        composite_feature<ilat::pair_fst, sparse_vec> make_feat(
            feat_dim_alloc& alloc,
            std::vector<std::string> features,
            std::vector<std::vector<double>> const& frames,
            std::unordered_map<std::string, std::string> const& args);

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
        
            pair_scrf graph;
            std::shared_ptr<pair_scrf> graph_path;
        
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
        
            ilat::fst ground_truth_fst;
        
            feat_dim_alloc gold_alloc;
        
            pair_scrf ground_truth;
            std::shared_ptr<pair_scrf> gold;
        
            std::shared_ptr<seg_cost<ilat::fst>> cost;
        
            learning_sample(learning_args const& args);
        };
        
        learning_args parse_learning_args(
            std::unordered_map<std::string, std::string> const& args);

        void make_gold(learning_sample& s, learning_args const& l_args);
        void make_min_cost_gold(learning_sample& s, learning_args const& l_args);
    }

}

#endif
