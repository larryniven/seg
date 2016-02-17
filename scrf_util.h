#ifndef SCRF_UTIL_H
#define SCRF_UTIL_H

#include "fst.h"
#include "scrf/scrf.h"
#include "scrf/lm.h"
#include <memory>
#include "nn/nn.h"

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

}


#endif
