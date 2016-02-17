#ifndef MAKE_FEAT_H
#define MAKE_FEAT_h

#include "scrf_feat.h"

namespace scrf {

    composite_feature make_feat(
        std::vector<std::string> features,
        std::vector<std::vector<real>> const& frames,
        std::unordered_map<std::string, int> const& phone_id);

}


#endif
