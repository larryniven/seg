#ifndef SCRF_UTIL_H
#define SCRF_UTIL_H

#include "fst.h"
#include "scrf/scrf.h"
#include "scrf/lm.h"
#include "scrf/scrf_cost.h"
#include <memory>

namespace scrf {

    std::shared_ptr<lm::fst> erase_input(std::shared_ptr<lm::fst> lm);

    lattice::fst make_segmentation_lattice(int frames, int min_seg_len, int max_seg_len);

    fst::path<scrf::scrf_t> make_min_cost_path(
        scrf::scrf_t& min_cost,
        fst::path<scrf::scrf_t> const& gold_path);
    
    scrf_t make_gold_scrf(lattice::fst gold,
        std::shared_ptr<lm::fst> lm);

    fst::path<scrf::scrf_t> make_ground_truth_path(
        scrf::scrf_t& ground_truth);

    scrf_t make_graph_scrf(int frames,
        std::shared_ptr<lm::fst> lm, int min_seg, int max_seg);

    scrf::scrf_t make_lat_scrf(lattice::fst lat, std::shared_ptr<lm::fst> lm);

    scrf_t make_forced_alignment_scrf(int frames,
        std::vector<std::string> const& labels, int min_seg_len, int max_seg_len);

    std::unordered_map<std::string, int> load_phone_id(std::string filename);

    namespace first_order {

        fst::path<scrf_t> make_min_cost_path(
            scrf_t& min_cost,
            fst::path<scrf_t> const& gold_path,
            std::vector<int> sils);

        fst::path<scrf_t> make_ground_truth_path(
            scrf_t& ground_truth);

        scrf_t make_graph_scrf(int frames,
            std::vector<int> const& labels, int min_seg, int max_seg);

        struct inference_args {
            int min_seg;
            int max_seg;
            scrf::first_order::param_t param;
            std::unordered_map<std::string, int> label_id;
            std::vector<std::string> id_label;
            std::vector<int> labels;
            std::vector<std::string> features;
            std::unordered_map<std::string, std::string> args;
        };
        
        struct sample {
            std::vector<std::vector<double>> frames;
        
            scrf::first_order::feat_dim_alloc graph_alloc;
        
            scrf::first_order::scrf_t graph;
            fst::path<scrf::first_order::scrf_t> graph_path;
        
            sample(inference_args const& args);
        };
        
        void make_graph(sample& s, inference_args const& i_args);
        
        struct learning_args
            : public inference_args {
        
            scrf::first_order::param_t opt_data;
            double step_size;
            double momentum;
            std::vector<int> sils;
        };
        
        struct learning_sample
            : public sample {
        
            ilat::fst ground_truth_fst;
        
            scrf::first_order::feat_dim_alloc gold_alloc;
        
            scrf::first_order::scrf_t ground_truth;
            fst::path<scrf::first_order::scrf_t> ground_truth_path;
        
            scrf::first_order::scrf_t gold;
            fst::path<scrf::first_order::scrf_t> gold_path;
        
            std::shared_ptr<scrf::first_order::seg_cost> cost;
        
            learning_sample(learning_args const& args);
        };
        
        learning_args parse_learning_args(
            std::unordered_map<std::string, std::string> const& args);
        
        void make_gold(learning_sample& s, learning_args const& l_args);
        void make_min_cost_gold(learning_sample& s, learning_args const& l_args);

    }

}


#endif
