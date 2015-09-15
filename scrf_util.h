#ifndef SCRF_UTIL_H
#define SCRF_UTIL_H

#include "fst.h"
#include "scrf/scrf.h"
#include "scrf/lm.h"
#include <memory>

namespace scrf {

    std::shared_ptr<lm::fst> erase_input(std::shared_ptr<lm::fst> lm);

    fst::path<scrf::scrf_t> make_min_cost_path(
        scrf::scrf_t& min_cost,
        fst::path<scrf::scrf_t> const& gold_path);
    
    fst::path<scrf::scrf_t> make_ground_truth_path(
        scrf::scrf_t& ground_truth);

    scrf_t make_graph_scrf(int frames,
        std::shared_ptr<lm::fst> lm, int min_seg, int max_seg);

    scrf::scrf_t make_lat_scrf(lattice::fst lat, std::shared_ptr<lm::fst> lm);
}


#endif
