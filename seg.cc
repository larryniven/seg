#include "seg/seg.h"

namespace fst {

    int edge_trait<int>::null = -1;

    std::tuple<int, int> edge_trait<std::tuple<int, int>>::null = std::make_tuple(-1, -1);

}
