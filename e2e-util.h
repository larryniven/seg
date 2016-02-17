#ifndef E2E_UTIL_H
#define E2E_UTIL_H

#include "nn/nn.h"

namespace scrf {

    std::vector<std::vector<double>> nn_feedforward(
        std::vector<std::vector<double>> const& frames,
        nn::nn_t const& nn);

}

#endif
