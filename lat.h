#ifndef LAT_H
#define LAT_H

#include "fst/ifst.h"

namespace lat {

    /*
     * Only acceptors can be loaded as of now.
     *
     */
    ifst::fst load_lattice(std::istream& is,
        std::unordered_map<std::string, int> const& symbol_id);

}

#endif
