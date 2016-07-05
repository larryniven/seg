#ifndef FSCRF_H
#define FSCRF_H

#include "scrf/scrf.h"
#include "scrf/scrf_weight.h"
#include "scrf/segcost.h"
#include "scrf/scrf_cost.h"
#include "autodiff/autodiff.h"
#include "nn/tensor_tree.h"

namespace fscrf {

    struct fscrf_data {
        std::shared_ptr<ilat::fst> fst;
        std::shared_ptr<std::vector<int>> topo_order;
        std::shared_ptr<scrf::scrf_weight<ilat::fst>> weight_func;
        std::shared_ptr<scrf::scrf_weight<ilat::fst>> cost_func;
        std::shared_ptr<tensor_tree::vertex> param;
    };

    using fscrf_fst = scrf::scrf_fst<fscrf_data>;

    struct fscrf_pair_data {
        std::shared_ptr<ilat::pair_fst> fst;
        std::shared_ptr<std::vector<std::tuple<int, int>>> topo_order;
        std::shared_ptr<scrf::scrf_weight<ilat::pair_fst>> weight_func;
        std::shared_ptr<scrf::scrf_weight<ilat::pair_fst>> cost_func;
        std::shared_ptr<tensor_tree::vertex> param;
    };

    using fscrf_pair_fst = scrf::scrf_fst<fscrf_pair_data>;

}

namespace scrf {

    template <>
    struct scrf_data_trait<fscrf::fscrf_data> {
        using base_fst = ilat::fst;
        using path_maker = ilat::ilat_path_maker;
        using edge = int;
        using vertex = int;
        using symbol = int;
        using fst = scrf_fst<fscrf::fscrf_data>;
    };

    template <>
    struct scrf_data_trait<fscrf::fscrf_pair_data> {
        using base_fst = ilat::pair_fst;
        using path_maker = ilat::pair_fst_path_maker;
        using edge = std::tuple<int, int>;
        using vertex = std::tuple<int, int>;
        using symbol = std::tuple<int, int>;
        using fst = scrf_fst<fscrf::fscrf_pair_data>;
    };

}

namespace fscrf {

    std::shared_ptr<ilat::fst> make_graph(int frames,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label,
        int min_seg_len, int max_seg_len, int stride);

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(
        std::vector<std::string> const& features);

    std::shared_ptr<scrf::composite_weight<ilat::fst>> make_weights(
        std::vector<std::string> const& features,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> frame_mat);

    struct frame_avg_score
        : public scrf::scrf_weight<ilat::fst> {

        std::shared_ptr<autodiff::op_t> param;
        std::shared_ptr<autodiff::op_t> frames;
        std::shared_ptr<autodiff::op_t> score;

        frame_avg_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames);

        virtual double operator()(ilat::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ilat::fst const& f,
            int e) const override;

        virtual void grad() const override;

    };

    struct frame_samples_score
        : public scrf::scrf_weight<ilat::fst> {

        std::shared_ptr<autodiff::op_t> param;
        std::shared_ptr<autodiff::op_t> frames;
        std::shared_ptr<autodiff::op_t> score;
        double scale;

        frame_samples_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames, double scale);

        virtual double operator()(ilat::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ilat::fst const& f,
            int e) const override;

        virtual void grad() const override;

    };

    struct left_boundary_score
        : public scrf::scrf_weight<ilat::fst> {

        std::shared_ptr<autodiff::op_t> param;
        std::shared_ptr<autodiff::op_t> frames;
        std::shared_ptr<autodiff::op_t> score;
        int shift;

        left_boundary_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames, int shift);

        virtual double operator()(ilat::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ilat::fst const& f,
            int e) const override;

        virtual void grad() const override;

    };

    struct right_boundary_score
        : public scrf::scrf_weight<ilat::fst> {

        std::shared_ptr<autodiff::op_t> param;
        std::shared_ptr<autodiff::op_t> frames;
        std::shared_ptr<autodiff::op_t> score;
        int shift;

        right_boundary_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames, int shift);

        virtual double operator()(ilat::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ilat::fst const& f,
            int e) const override;

        virtual void grad() const override;

    };

    struct log_length_score
        : public scrf::scrf_weight<ilat::fst> {

        std::shared_ptr<autodiff::op_t> param;

        log_length_score(std::shared_ptr<autodiff::op_t> param);

        virtual double operator()(ilat::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ilat::fst const& f,
            int e) const override;

    };

    struct length_score
        : public scrf::scrf_weight<ilat::fst> {

        std::shared_ptr<autodiff::op_t> param;

        length_score(std::shared_ptr<autodiff::op_t> param);

        virtual double operator()(ilat::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ilat::fst const& f,
            int e) const override;

    };

    struct bias_score
        : public scrf::scrf_weight<ilat::fst> {

        std::shared_ptr<autodiff::op_t> param;

        bias_score(std::shared_ptr<autodiff::op_t> param);

        virtual double operator()(ilat::fst const& f,
            int e) const;

        virtual void accumulate_grad(double g, ilat::fst const& f,
            int e) const override;

    };

    struct inference_args {
        int min_seg;
        int max_seg;
        int stride;
        std::shared_ptr<tensor_tree::vertex> param;
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
        fscrf_data graph_data;

        sample(inference_args const& i_args);
    };

    void make_graph(sample& s, inference_args const& i_args);

    struct learning_args
        : public inference_args {

        double cost_scale;
        std::shared_ptr<tensor_tree::vertex> opt_data;
        double l2;
        double step_size;
        double momentum;
        double decay;
        std::vector<int> sils;
    };

    void parse_learning_args(learning_args& l_args,
        std::unordered_map<std::string, std::string> const& args);

    struct learning_sample
        : public sample {

        std::vector<segcost::segment<int>> gt_segs;
        fscrf_data gold_data;

        learning_sample(learning_args const& args);
    };

    struct hinge_loss {

        fscrf_data& graph_data;

        fscrf_data graph_path_data;
        fscrf_data gold_path_data;

        std::vector<int> const& sils;
        std::vector<segcost::segment<int>> gold_segs;
        double cost_scale;

        hinge_loss(fscrf_data& graph_data,
            std::vector<segcost::segment<int>> const& gt_segs,
            std::vector<int> const& sils,
            double cost_scale);

        double loss() const;

        void grad() const;

    };

    struct log_loss {

        log_loss(fscrf_data& graph_data);

        double loss() const;

        void grad() const;

    };

    struct mode2_weight
        : public scrf::scrf_weight<ilat::pair_fst> {

        std::shared_ptr<scrf::scrf_weight<ilat::fst>> weight;

        virtual double operator()(ilat::pair_fst const& fst,
            std::tuple<int, int> e) const override;

        virtual void accumulate_grad(double g, ilat::pair_fst const& fst,
            std::tuple<int, int> e) const override;

        virtual void grad() const override;
    };
}

#endif
