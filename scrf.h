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

namespace weiran {

    struct nn_t;

}

namespace scrf {

    struct param_t {
        std::unordered_map<std::string, std::vector<real>> class_param;
    };

    param_t& operator-=(param_t& p1, param_t const& p2);
    param_t& operator+=(param_t& p1, param_t const& p2);
    param_t& operator*=(param_t& p1, real c);

    param_t load_param(std::istream& is);
    param_t load_param(std::string filename);

    void save_param(std::ostream& os, param_t const& param);
    void save_param(std::string filename, param_t const& param);

    void adagrad_update(param_t& theta, param_t const& grad,
        param_t& accu_grad_sq, real step_size);

    struct scrf_feature {

        virtual ~scrf_feature();

        virtual int size() const = 0;

        virtual void operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const = 0;

    };

    struct composite_feature
        : public scrf_feature {

        std::vector<std::shared_ptr<scrf_feature>> features;

        virtual int size() const override;

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

            virtual int size() const override;

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;

        };

        struct length_value
            : public scrf_feature {

            int max_seg;

            length_value(int max_seg);

            virtual int size() const override;

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct length_indicator
            : public scrf_feature {

            int max_seg;

            length_indicator(int max_seg);

            virtual int size() const override;

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct frame_avg
            : public scrf_feature {

            std::vector<std::vector<real>> const& inputs;
            int start_dim;
            int end_dim;

            mutable std::unordered_map<int, std::vector<real>> feat_cache;

            frame_avg(std::vector<std::vector<real>> const& inputs,
                int start_dim = -1, int end_dim = -1);

            virtual int size() const override;

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct frame_samples
            : public scrf_feature {

            std::vector<std::vector<real>> const& inputs;
            int samples;
            int start_dim;
            int end_dim;

            frame_samples(std::vector<std::vector<real>> const& inputs, int samples,
                int start_dim = -1, int end_dim = -1);

            virtual int size() const override;

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct left_boundary
            : public scrf_feature {

            std::vector<std::vector<real>> const& inputs;
            int start_dim;
            int end_dim;

            mutable std::unordered_map<int, std::vector<real>> feat_cache;

            left_boundary(std::vector<std::vector<real>> const& inputs,
                int start_dim = -1, int end_dim = -1);

            virtual int size() const override;

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct right_boundary
            : public scrf_feature {

            std::vector<std::vector<real>> const& inputs;
            int start_dim;
            int end_dim;

            mutable std::unordered_map<int, std::vector<real>> feat_cache;

            right_boundary(std::vector<std::vector<real>> const& inputs,
                int start_dim = -1, int end_dim = -1);

            virtual int size() const override;

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct lm_score
            : public scrf_feature {

            virtual int size() const override;

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
        };

        struct lattice_score
            : public scrf_feature {

            virtual int size() const override;

            virtual void operator()(
                param_t& feat,
                fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;

        };

    }

    namespace score {

        struct linear_score
            : public scrf_weight {

            param_t const& param;
            std::shared_ptr<scrf_feature> feat;

            linear_score(param_t const& param, std::shared_ptr<scrf_feature> feat);

            virtual real operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;

        };

        struct label_score
            : public scrf_weight {

            param_t const& param;
            std::shared_ptr<scrf_feature> feat;

            mutable std::unordered_map<std::string, real> cache;

            label_score(param_t const& param,
                std::shared_ptr<scrf_feature> feat);

            virtual real operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
            
        };

        struct lm_score
            : public scrf_weight {

            param_t const& param;
            std::shared_ptr<scrf_feature> feat;

            mutable std::unordered_map<int, real> cache;

            lm_score(param_t const& param,
                std::shared_ptr<scrf_feature> feat);

            virtual real operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
            
        };

        struct lattice_score
            : public scrf_weight {

            param_t const& param;
            std::shared_ptr<scrf_feature> feat;

            mutable std::unordered_map<std::tuple<int, std::string>, real> cache;

            lattice_score(param_t const& param,
                std::shared_ptr<scrf_feature> feat);

            virtual real operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
                std::tuple<int, int> const& e) const override;
            
        };

    }

    std::vector<std::tuple<int, int>> topo_order(scrf_t const& scrf);

    fst::path<scrf_t> shortest_path(scrf_t const& s,
        std::vector<std::tuple<int, int>> const& order);

    lattice::fst load_gold(std::istream& is);

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

    struct loss_func {

        virtual ~loss_func();

        virtual real loss() = 0;
        virtual param_t param_grad() = 0;

    };

    struct hinge_loss
        : public loss_func {

        fst::path<scrf_t> const& gold;
        scrf_t const& graph;
        fst::path<scrf_t> graph_path;

        hinge_loss(fst::path<scrf_t> const& gold, scrf_t const& graph);

        virtual real loss() override;
        virtual param_t param_grad() override;
    };

    struct filtering_loss
        : public loss_func {

        fst::path<scrf_t> const& gold;
        scrf_t const& graph;
        real alpha;

        fst::path<scrf_t> graph_path;
        fst::forward_one_best<scrf_t> forward;
        fst::backward_one_best<scrf_t> backward;
        std::unordered_map<scrf_t::vertex_type, param_t> f_param;
        std::unordered_map<scrf_t::vertex_type, param_t> b_param;
        real threshold;

        filtering_loss(
            fst::path<scrf_t> const& gold,
            scrf_t const& graph,
            real alpha);

        virtual real loss() override;
        virtual param_t param_grad() override;
    };

    struct hinge_loss_beam
        : public loss_func {

        fst::path<scrf_t> const& gold;
        scrf_t const& graph;
        fst::path<scrf_t> graph_path;
        int beam_width;

        hinge_loss_beam(fst::path<scrf_t> const& gold, scrf_t const& graph, int beam_width);

        virtual real loss() override;
        virtual param_t param_grad() override;
    };

    composite_feature make_feature(
        std::vector<std::string> features,
        std::vector<std::vector<real>> const& inputs, int max_seg);

    composite_feature make_feature(
        std::vector<std::string> features,
        std::vector<std::vector<real>> const& inputs, int max_seg,
        std::vector<real> const& cm_mean, std::vector<real> const& cm_stddev,
        weiran::nn_t const& nn);

    composite_weight make_weight(
        param_t const& param,
        std::vector<std::string> features,
        composite_feature const& feat);

    lattice::fst make_lattice(
        std::vector<std::vector<real>> acoustics,
        std::unordered_set<std::string> phone_set,
        int seg_size);

}



#endif
