#ifndef SCRF_H
#define SCRF_H

#include <vector>
#include <string>
#include <tuple>
#include <limits>
#include <ostream>
#include <iostream>
#include <unordered_map>
#include <map>
#include "ebt/ebt.h"
#include "scrf/fst.h"
#include "scrf/lm.h"
#include "scrf/lattice.h"

namespace scrf {

    struct param_t {
        std::unordered_map<std::string, std::vector<real>> class_param;
    };

    param_t& operator-=(param_t& p1, param_t const& p2);
    param_t& operator+=(param_t& p1, param_t const& p2);

    param_t load_param(std::istream& is);
    param_t load_param(std::string filename);

    void save_param(std::ostream& os, param_t const& param);
    void save_param(std::string filename, param_t const& param);

    void adagrad_update(param_t& theta, param_t const& grad,
        param_t& accu_grad_sq, real step_size);

    struct scrf_feature {

        virtual ~scrf_feature();

        virtual void operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const = 0;

    };

    struct composite_feature
        : public scrf_feature {

        std::vector<std::shared_ptr<scrf_feature>> features;

        virtual void operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const override;
        
    };

    struct scrf_weight {

        virtual ~scrf_weight();

        virtual real operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const = 0;

    };

    struct composite_weight
        : public scrf_weight {

        std::vector<std::shared_ptr<scrf_weight>> weights;

        virtual real operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const override;

    };

    struct linear_score
        : public scrf_weight {

        param_t const& param;
        scrf_feature const& feat_func;

        linear_score(param_t const& param, scrf_feature const& feat_func);

        virtual real operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const override;

    };

    struct scrf_t {
        using fst_type = fst::composed_fst<lattice::fst, lm::fst>;
        using vertex_type = fst_type::vertex_type;
        using edge_type = fst_type::edge_type;

        std::shared_ptr<fst_type> fst;
        std::shared_ptr<scrf_weight> weight_func;
        std::shared_ptr<scrf_feature> feature_func;

        std::vector<vertex_type> topo_order;

        std::vector<vertex_type> vertices() const;
        std::vector<edge_type> edges() const;
        vertex_type head(edge_type const& e) const;
        vertex_type tail(edge_type const& e) const;
        std::vector<edge_type> in_edges(vertex_type const& v) const;
        std::vector<edge_type> out_edges(vertex_type const& v) const;
        real weight(edge_type const& e) const;
        std::string input(edge_type const& e) const;
        std::string output(edge_type const& e) const;
        std::vector<vertex_type> initials() const;
        std::vector<vertex_type> finals() const;
    };

    namespace feature {

        struct bias
            : public scrf_feature {

            bias();

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;

        };

        struct length_value
            : public scrf_feature {

            int max_seg;

            length_value(int max_seg);

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct length_indicator
            : public scrf_feature {

            int max_seg;

            length_indicator(int max_seg);

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct frame_avg
            : public scrf_feature {

            std::vector<std::vector<real>> const& inputs;

            mutable std::unordered_map<int, std::vector<real>> feat_cache;

            frame_avg(std::vector<std::vector<real>> const& inputs);

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct frame_samples
            : public scrf_feature {

            std::vector<std::vector<real>> const& inputs;
            int samples;

            frame_samples(std::vector<std::vector<real>> const& inputs, int samples);

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct left_boundary
            : public scrf_feature {

            std::vector<std::vector<real>> const& inputs;

            mutable std::unordered_map<int, std::vector<real>> feat_cache;

            left_boundary(std::vector<std::vector<real>> const& inputs);

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct right_boundary
            : public scrf_feature {

            std::vector<std::vector<real>> const& inputs;

            mutable std::unordered_map<int, std::vector<real>> feat_cache;

            right_boundary(std::vector<std::vector<real>> const& inputs);

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct lm_score
            : public scrf_feature {

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct lattice_score
            : public scrf_feature {

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;

        };

    }

    fst::path<scrf_t> shortest_path(scrf_t& s,
        std::vector<std::tuple<int, int>> const& order);

    lattice::fst load_gold(std::istream& is);

    std::vector<std::vector<real>> load_features(std::string filename);
    std::vector<std::vector<real>> load_features(std::string filename, int nfeat);

    std::unordered_set<std::string> load_phone_set(std::string filename);

    scrf_t make_gold_scrf(lattice::fst gold,
        std::shared_ptr<lm::fst> lm);

    lattice::fst make_segmentation_lattice(int frames, int max_seg);

    std::shared_ptr<lm::fst> erase_input(std::shared_ptr<lm::fst> lm);

    struct backoff_cost
        : public scrf_weight {

        virtual real operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const;
    };

    struct overlap_cost
        : public scrf_weight {

        fst::path<scrf_t> const& gold;

        mutable std::unordered_map<int, std::vector<std::tuple<int, int>>> edge_cache;

        overlap_cost(fst::path<scrf_t> const& gold);

        virtual real operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const;

    };

    struct neg_cost
        : public scrf_weight {

        std::shared_ptr<scrf_weight> cost;

        neg_cost(std::shared_ptr<scrf_weight> cost);

        virtual real operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const;
    };

    scrf_t make_graph_scrf(int frames,
        std::shared_ptr<lm::fst> lm, int max_seg);

    struct hinge_loss {
        fst::path<scrf_t> const& gold;
        scrf_t& graph;
        fst::path<scrf_t> graph_path;

        hinge_loss(fst::path<scrf_t> const& gold, scrf_t& graph);

        real loss();
        param_t param_grad();
    };

    struct filtering_loss {
        fst::path<scrf_t> const& gold;
        scrf_t& graph;
        fst::path<scrf_t> graph_path;

        filtering_loss(fst::path<scrf_t> const& gold, scrf_t& graph);

        real loss();
        param_t param_grad();
    };

    composite_feature make_feature(
        std::vector<std::string> features,
        std::vector<std::vector<real>> const& inputs, int max_seg);

    lattice::fst make_lattice(
        std::vector<std::vector<real>> acoustics,
        std::unordered_set<std::string> phone_set,
        int seg_size);

#if 0
    struct forward_backward_alg {

        std::unordered_map<std::tuple<int, int>, real> alpha;
        std::unordered_map<std::tuple<int, int>, real> beta;

        void forward_score(scrf const& s);
        void backward_score(scrf const& s);

        std::unordered_map<std::string, std::vector<real>> feature_expectation(scrf const& s);
    };

    struct log_loss {

        fst::path<scrf> const& gold;
        scrf const& lat;
        forward_backward_alg fb;

        std::unordered_map<std::string, std::vector<real>> result;

        log_loss(fst::path<scrf> const& gold, scrf const& lat);

        real loss();
        std::unordered_map<std::string, std::vector<real>> const& model_grad();

    };

    struct frame_feature {
        std::vector<std::vector<real>> const& inputs;

        mutable std::unordered_map<std::tuple<int, int>,
            std::unordered_map<std::string, std::vector<real>>> cache;

        frame_feature(std::vector<std::vector<real>> const& inputs);

        std::unordered_map<std::string, std::vector<real>> const&
        operator()(std::string const& y, int start_time, int end_time) const;
    };

    struct frame_score {
        frame_feature const& feat;
        param_t const& model;

        mutable std::unordered_map<std::tuple<std::string, int, int>, real> cache;

        frame_score(frame_feature const& feat, param_t const& model);

        real operator()(std::string const& y, int start_time, int end_time) const;
    };

    struct linear_score
        : public scrf_weight {

        fst::composed_fst<lattice::fst, lm::fst> const& fst;
        frame_score const& f_score;

        linear_score(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            frame_score const& f_score);

        virtual real operator()(std::tuple<int, int> const& e) const override;

    };
#endif

}



#endif
