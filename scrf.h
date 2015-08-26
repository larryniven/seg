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

        virtual std::string name() const = 0;

        virtual void operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const = 0;

    };

    struct collapsed_feature
        : public scrf_feature {

        std::shared_ptr<scrf_feature> feat_func;
        std::string name_;

        collapsed_feature(std::string name,
            std::shared_ptr<scrf_feature> feat_func);

        virtual int size() const override;

        virtual std::string name() const override;

        virtual void operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const override;

    };

    struct composite_feature
        : public scrf_feature {

        std::vector<std::shared_ptr<scrf_feature>> features;

        composite_feature(std::string name);

        virtual int size() const override;

        virtual std::string name() const override;

        virtual void operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const override;

    private:
        std::string name_;
        
    };

    struct scrf_weight {

        virtual ~scrf_weight();

        virtual real operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const = 0;

    };

    struct linear_weight
        : public scrf_weight {

        param_t const& param;
        scrf_feature const& feat_func;

        linear_weight(param_t const& param, scrf_feature const& feat_func);

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

    struct loss_func {

        virtual ~loss_func();

        virtual real loss() = 0;
        virtual param_t param_grad() = 0;

    };

    lattice::fst make_lattice(
        std::vector<std::vector<real>> acoustics,
        std::unordered_set<std::string> phone_set,
        int seg_size);

}

#endif
