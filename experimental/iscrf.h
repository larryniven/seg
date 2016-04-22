#ifndef ISCRF_H
#define ISCRF_H

#include "scrf/experimental/scrf.h"
#include "scrf/experimental/scrf_feat.h"
#include "scrf/experimental/scrf_cost.h"
#include "scrf/experimental/scrf_weight.h"
#include "nn/lstm.h"
#include "nn/pred.h"

namespace scrf {

    struct iscrf
        : public scrf<int, int, int, dense_vec> {

        using base_fst = ilat::fst;
        using vector = dense_vec;
        using vertex = int;
        using edge = int;
        using symbol = int;

        std::shared_ptr<ilat::fst> fst;
        std::vector<vertex> topo_order_cache;

        std::shared_ptr<scrf_weight<ilat::fst>> weight_func;
        std::shared_ptr<scrf_feature<ilat::fst, dense_vec>> feature_func;
        std::shared_ptr<scrf_weight<ilat::fst>> cost_func;

        virtual std::vector<int> const& vertices() const override;
        virtual std::vector<int> const& edges() const override;
        virtual int head(int e) const override;
        virtual int tail(int e) const override;
        virtual std::vector<int> const& in_edges(int v) const override;
        virtual std::vector<int> const& out_edges(int v) const override;
        virtual double weight(int e) const override;
        virtual int const& input(int e) const override;
        virtual int const& output(int e) const override;
        virtual std::vector<int> const& initials() const override;
        virtual std::vector<int> const& finals() const override;

        virtual long time(int v) const override;

        virtual void feature(dense_vec& f, int e) const override;
        virtual double cost(int e) const override;

        virtual std::vector<int> const& topo_order() const override;

    };

    struct iscrf_path_maker
        : public fst::path_maker<iscrf> {

        virtual std::shared_ptr<iscrf> operator()(std::vector<int> const& edges,
            iscrf const& f) const override;
    };

    std::shared_ptr<ilat::fst> make_graph(int frames,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label,
        int min_seg_len, int max_seg_len);

    struct ilat_lexicalizer
        : public lexicalizer<ilat::fst, dense_vec> {

        virtual double* lex(feat_dim_alloc const& alloc,
            int order, dense_vec& f, ilat::fst const& a, int e) const override;
    };

    std::pair<int, int> get_dim(std::string feat);

    composite_feature<ilat::fst, dense_vec> make_feat(
        feat_dim_alloc& alloc,
        std::vector<std::string> features,
        std::vector<std::vector<double>> const& frames,
        std::unordered_map<std::string, std::string> const& args);

    struct inference_args {
        int min_seg;
        int max_seg;
        dense_vec param;
        std::unordered_map<std::string, int> label_id;
        std::vector<std::string> id_label;
        std::vector<int> labels;
        std::vector<std::string> features;
        std::unordered_map<std::string, std::string> args;
    };

    void parse_inference_args(inference_args& l_args,
        std::unordered_map<std::string, std::string> const& args);

    struct sample {
        std::vector<std::vector<double>> frames;

        feat_dim_alloc graph_alloc;

        iscrf graph;
        std::shared_ptr<iscrf> graph_path;

        sample(inference_args const& i_args);
    };

    void make_graph(sample& s, inference_args const& i_args);

    void make_lattice(ilat::fst const& lat, sample& s, inference_args const& i_args);

    struct learning_args
        : public inference_args {

        double cost_scale;
        dense_vec opt_data;
        double step_size;
        double momentum;
        double decay;
        std::vector<int> sils;
    };

    struct learning_sample
        : public sample {

        std::vector<segcost::segment<int>> gold_segs;

        feat_dim_alloc gold_alloc;

        std::shared_ptr<iscrf> gold;

        std::shared_ptr<scrf_weight<ilat::fst>> cost;

        learning_sample(learning_args const& args);
    };

    std::vector<segcost::segment<int>> load_segments(std::istream& is,
        std::unordered_map<std::string, int> const& label_id);

    std::vector<segcost::segment<std::string>> load_segments(std::istream& is);

    void parse_learning_args(learning_args& l_args,
        std::unordered_map<std::string, std::string> const& args);

    void make_min_cost_gold(learning_sample& s, learning_args const& l_args);

    void parameterize(iscrf& scrf, feat_dim_alloc& alloc,
        std::vector<std::vector<double>> const& frames,
        inference_args const& i_args);

    void parameterize(learning_sample& s, learning_args const& l_args);

    ilat::fst make_label_seq_fst(std::vector<std::string> const& label_seq,
        std::unordered_map<std::string, int> const& label_id);

}


#endif
