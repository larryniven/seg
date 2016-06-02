#ifndef ISCRF_H
#define ISCRF_H

#include "scrf/experimental/scrf.h"
#include "scrf/experimental/scrf_feat.h"
#include "scrf/experimental/scrf_cost.h"
#include "scrf/experimental/scrf_weight.h"
#include "nn/lstm.h"
#include "nn/pred.h"

namespace iscrf {

    struct iscrf_data {
        std::shared_ptr<ilat::fst> fst;
        std::shared_ptr<std::vector<int>> topo_order;
        std::shared_ptr<scrf::scrf_weight<ilat::fst>> weight_func;
        std::shared_ptr<scrf::composite_feature<ilat::fst, scrf::dense_vec>> feature_func;
        std::shared_ptr<scrf::scrf_weight<ilat::fst>> cost_func;
        std::shared_ptr<scrf::dense_vec> param;
        std::shared_ptr<std::vector<std::string>> features;
    };

    using iscrf_fst = scrf::scrf_fst<iscrf_data>;

}

namespace scrf {

    template <>
    struct scrf_data_trait<iscrf::iscrf_data> {
        using base_fst = ilat::fst;
        using path_maker = ilat::ilat_path_maker;
        using edge = int;
        using vertex = int;
        using symbol = int;
        using fst = scrf_fst<iscrf::iscrf_data>;
        using vector = dense_vec;
    };

}

namespace iscrf {

    double weight(iscrf_data const& data, int e);
    void feature(iscrf_data const& data, scrf::dense_vec& f, int e);
    double cost(iscrf_data const& data, int e);

    std::shared_ptr<ilat::fst> make_graph(int frames,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label,
        int min_seg_len, int max_seg_len, int stride);

    struct ilat_lexicalizer
        : public scrf::lexicalizer<ilat::fst, scrf::dense_vec> {

        virtual double* lex(scrf::feat_dim_alloc const& alloc,
            int order, scrf::dense_vec& f, ilat::fst const& a, int e) const override;
    };

    scrf::composite_feature<ilat::fst, scrf::dense_vec> make_feat(
        scrf::feat_dim_alloc& alloc,
        std::vector<std::string> features,
        std::vector<std::vector<double>> const& frames,
        std::unordered_map<std::string, std::string> const& args);

    struct inference_args {
        int min_seg;
        int max_seg;
        int stride;
        scrf::dense_vec param;
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

        scrf::feat_dim_alloc graph_alloc;

        iscrf_data graph_data;

        sample(inference_args const& i_args);
    };

    void make_graph(sample& s, inference_args const& i_args);

    void make_lattice(ilat::fst const& lat, sample& s, inference_args const& i_args);

    struct learning_args
        : public inference_args {

        double cost_scale;
        scrf::dense_vec opt_data;
        double step_size;
        double momentum;
        double decay;
        std::vector<int> sils;
    };

    struct learning_sample
        : public sample {

        std::vector<segcost::segment<int>> gold_segs;

        scrf::feat_dim_alloc gold_alloc;

        iscrf_data gold_data;

        learning_sample(learning_args const& args);
    };

    std::vector<segcost::segment<int>> load_segments(std::istream& is,
        std::unordered_map<std::string, int> const& label_id);

    std::vector<segcost::segment<std::string>> load_segments(std::istream& is);

    void parse_learning_args(learning_args& l_args,
        std::unordered_map<std::string, std::string> const& args);

    void make_min_cost_gold(learning_sample& s, learning_args const& l_args);

    void parameterize(iscrf_data& data, scrf::feat_dim_alloc& alloc,
        std::vector<std::vector<double>> const& frames,
        inference_args const& i_args);

    void parameterize(learning_sample& s, learning_args const& l_args);

}

#endif
