#include "scrf/segcost.h"
#include <cassert>
#include <iostream>

namespace segcost {

    double overlap_cost::operator()(std::vector<speech::segment> const& gold_segs,
        speech::segment const& e) const
    {
        if (e.label == "<eps>" && e.start_time == e.end_time) {
            return 0;
        }

        speech::segment max_seg;
        int max_overlap = -1;

        for (auto& s: gold_segs) {
            int overlap = std::max<int>(0, std::min(s.end_time, e.end_time)
                - std::max(s.start_time, e.start_time));

            if (overlap > max_overlap) {
                max_seg = s;
                max_overlap = overlap;
            }
        }

        int union_ = std::max(max_seg.end_time, e.end_time)
            - std::min(max_seg.start_time, e.start_time);

        if ((e.label == "<s>" || e.label == "</s>" || e.label == "sil") && e.label == max_seg.label) {
            assert(e.end_time - e.start_time - max_overlap >= 0);
            return (e.end_time - e.start_time) - max_overlap;
        } else if (e.label == max_seg.label) {
            return union_ - max_overlap;
        } else {
            return union_;
        }
    }

}

