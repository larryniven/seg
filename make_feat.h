#ifndef MAKE_FEAT_H
#define MAKE_FEAT_h

#include "scrf_feat.h"

namespace scrf {

    composite_feature make_feat(
        std::vector<std::string> features,
        std::vector<std::vector<real>> const& frames,
        std::unordered_map<std::string, std::vector<int>> const& label_dim);

    namespace first_order {

        composite_feature make_feat(
            feat_dim_alloc& alloc,
            std::vector<std::string> features,
            std::vector<std::vector<real>> const& frames,
            std::vector<std::vector<int>> const& label_dim);

    }

}


#endif
