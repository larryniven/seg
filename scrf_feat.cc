#include "scrf/scrf_feat.h"
#include <cassert>
#include <fstream>

namespace scrf {

    feat_dim_alloc::feat_dim_alloc(std::vector<int> const& labels)
        : labels(labels)
    {}

    int feat_dim_alloc::alloc(int order, int dim)
    {
        if (order >= order_dim.size()) {
            order_dim.resize(order + 1);
        }

        int result = order_dim[order];
        order_dim[order] += dim;

        return result;
    }

}
