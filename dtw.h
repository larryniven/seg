#ifndef DTW_H
#define DTW_H

#include "la/la.h"

namespace dtw {

    double l2_dist(std::vector<double> const& v1, std::vector<double> const& v2);
    double dtw(std::vector<std::vector<double>> const& seg1, std::vector<std::vector<double>> const& seg2);

}

#endif
