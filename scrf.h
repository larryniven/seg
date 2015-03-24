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

    struct scrf_model {
        std::unordered_map<std::string, std::vector<double>> weights;
    };

    scrf_model load_model(std::istream& is);
    scrf_model load_model(std::string filename);

    void save_model(std::ostream& os, scrf_model const& model);
    void save_model(std::string filename, scrf_model const& model);

    void adagrad(scrf_model& theta, scrf_model const& grad,
        scrf_model& accu_grad_sq, double step_size);

    struct scrf_weight {
        virtual ~scrf_weight();

        virtual double operator()(std::tuple<int, int> const& e) const = 0;
    };

    struct scrf {
        using fst_type = fst::composed_fst<lattice::fst, lm::fst>;
        using vertex_type = fst_type::vertex_type;
        using edge_type = fst_type::edge_type;

        std::shared_ptr<fst_type> fst;
        std::shared_ptr<scrf_weight> weight_func;

        std::vector<vertex_type> vertices() const;
        std::vector<edge_type> edges() const;
        vertex_type head(edge_type const& e) const;
        vertex_type tail(edge_type const& e) const;
        std::vector<edge_type> in_edges(vertex_type const& v) const;
        std::vector<edge_type> out_edges(vertex_type const& v) const;
        double weight(edge_type const& e) const;
        std::string input(edge_type const& e) const;
        std::string output(edge_type const& e) const;
        vertex_type initial() const;
        vertex_type final() const;
    };

    lattice::fst_data load_gold(std::istream& is);
    lattice::fst_data load_gold(std::istream& is, lattice::fst_data const& scrf_d);

    std::vector<std::vector<double>> load_acoustics(std::string filename);
    std::vector<std::vector<double>> load_acoustics(std::string filename, int nfeat);
    std::unordered_set<std::string> load_phone_set(std::string filename);

}

#endif
