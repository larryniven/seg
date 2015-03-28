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

namespace std {

    template <>
    struct hash<tuple<int, int>> {
        size_t operator()(tuple<int, int> const& t) const;
    };

}

namespace scrf {

    struct scrf_model {
        std::unordered_map<std::string, std::vector<real>> weights;
    };

    scrf_model load_model(std::istream& is);
    scrf_model load_model(std::string filename);

    void save_model(std::ostream& os, scrf_model const& model);
    void save_model(std::string filename, scrf_model const& model);

    void adagrad(scrf_model& theta, scrf_model const& grad,
        scrf_model& accu_grad_sq, real step_size);

    struct scrf_weight {
        virtual ~scrf_weight();

        virtual real operator()(std::tuple<int, int> const& e) const = 0;
    };

    struct scrf_feature {
        virtual ~scrf_feature();

        virtual std::unordered_map<std::string, std::vector<real>> const&
        operator()(std::tuple<int, int> const& e) const = 0;
    };

    struct scrf {
        using fst_type = fst::composed_fst<lattice::fst, lm::fst>;
        using vertex_type = fst_type::vertex_type;
        using edge_type = fst_type::edge_type;

        std::shared_ptr<fst_type> fst;
        std::shared_ptr<scrf_feature> feature_func;
        std::shared_ptr<scrf_weight> weight_func;
        std::vector<std::tuple<int, int>> topo_order;

        std::vector<vertex_type> vertices() const;
        std::vector<edge_type> edges() const;
        vertex_type head(edge_type const& e) const;
        vertex_type tail(edge_type const& e) const;
        std::vector<edge_type> in_edges(vertex_type const& v) const;
        std::vector<edge_type> out_edges(vertex_type const& v) const;
        real weight(edge_type const& e) const;
        std::string input(edge_type const& e) const;
        std::string output(edge_type const& e) const;
        vertex_type initial() const;
        vertex_type final() const;
    };

    fst::path<scrf> shortest_path(scrf& s,
        std::vector<std::tuple<int, int>> const& order);

    lattice::fst make_lattice(
        std::vector<std::vector<real>> acoustics,
        std::unordered_set<std::string> phone_set,
        int seg_size);

    lattice::fst load_gold(std::istream& is);
    lattice::fst load_gold(std::istream& is, lattice::fst_data const& scrf_d);

    std::vector<std::vector<real>> load_features(std::string filename);
    std::vector<std::vector<real>> load_features(std::string filename, int nfeat);

    std::unordered_set<std::string> load_phone_set(std::string filename);

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

    struct backoff_cost
        : public scrf_weight {

        fst::composed_fst<lattice::fst, lm::fst> const& fst;

        backoff_cost(fst::composed_fst<lattice::fst, lm::fst> const& fst);

        virtual real operator()(std::tuple<int, int> const& e) const;
    };

    std::shared_ptr<lm::fst> erase_input(std::shared_ptr<lm::fst> lm);

    scrf make_gold_scrf(lattice::fst gold,
        std::shared_ptr<lm::fst> lm);

    lattice::fst make_segmentation_lattice(int frames, int max_seg);

    scrf make_graph_scrf(int frames,
        std::shared_ptr<lm::fst> lm, int max_seg);

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
        scrf_model const& model;

        mutable std::unordered_map<std::tuple<std::string, int, int>, real> cache;

        frame_score(frame_feature const& feat, scrf_model const& model);

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

    namespace detail {

        struct model_vector {
            std::vector<std::vector<real>> class_weights;
        };

        void save_model(std::ostream& os, model_vector const& model);
        void save_model(std::string filename, model_vector const& model);

        model_vector load_model(std::istream& is);
        model_vector load_model(std::string filename);

        struct scrf {
            scrf(std::vector<std::vector<real>> const& inputs,
                model_vector const& weights, int labels);

            std::vector<std::vector<real>> const& inputs;
            model_vector const& weights;
            int labels;

            mutable std::vector<std::vector<real>> feat;

            virtual real score(int y, int start_time, int end_time) const;

            virtual real score(int y1, int y2, int start_time, int end_time) const;

            virtual std::vector<std::vector<real>> const&
            feature(int y, int start_time, int end_time) const;

        };

        struct gold_scrf
            : public scrf {

            struct edge {
                 int start_time;
                 int end_time;
                 int label;
            };

            std::vector<edge> edges;

            gold_scrf(std::vector<std::vector<real>> const& inputs,
                model_vector const& weights, int labels);

        };

        struct graph_scrf
            : public scrf {

            graph_scrf(std::vector<std::vector<real>> const& input,
                model_vector const& weights, int labels,
                std::unordered_map<std::string, int> const& phone_map,
                int frames, int max_seg);

            std::unordered_map<std::string, int> const& phone_map;
            int frames;
            int max_seg;

            mutable std::vector<std::vector<std::vector<real>>> score_cache;
            mutable std::vector<std::vector<std::vector<std::vector<std::vector<real>>>>> feature_cache;

            virtual real score(int y, int start_time, int end_time) const override;

            virtual real score(int y1, int y2, int start_time, int end_time) const override;

            virtual std::vector<std::vector<real>> const&
            feature(int y, int start_time, int end_time) const override;
        };

        struct forward_backward_alg {

            graph_scrf const& model;

            std::vector<std::vector<real>> alpha;
            std::vector<std::vector<real>> beta;

            void forward_score();
            void backward_score();

            std::vector<std::vector<real>> feature_expectation();

        };

        struct log_loss {

            gold_scrf const& gold;
            graph_scrf const& graph;
            forward_backward_alg fb;

            log_loss(gold_scrf const& gold, graph_scrf const& graph);
            real loss();
            std::vector<std::vector<real>> model_grad();

        };

        std::vector<gold_scrf::edge> load_gold(std::istream& is,
            std::unordered_map<std::string, int> const& phone_map,
            int frames);

        std::unordered_map<std::string, int>
        load_phone_map(std::string filename);

        std::vector<std::string>
        make_inv_phone_map(std::unordered_map<std::string, int> const& phone_map);

    }

}



#endif
