#include "scrf/e2e-util.h"
#include "la/la-gpu.h"
#include "autodiff/autodiff-gpu.h"

namespace scrf {

    std::vector<std::vector<double>> nn_feedforward(
        std::vector<std::vector<double>> const& frames,
        nn::nn_t const& nn)
    {
        int dim = frames.front().size();
    
        std::vector<std::vector<double>> result;

        la::vector<double> input_block;
        input_block.resize(frames.size() * 11 * dim);

        for (int i = 0; i < frames.size(); ++i) {
            std::vector<double> input;
    
            for (int j = i - 5; j <= i + 5; ++j) {
                if (j < 0 || j >= frames.size()) {
                    input.resize(input.size() + dim);
                } else {
                    input.insert(input.end(), frames[j].begin(), frames[j].end());
                }
            }

            std::copy(input.begin(), input.end(), input_block.data() + 11 * dim * i);
        }

        la::gpu::vector<double> input_gpu_block { input_block };

        for (int i = 0; i < frames.size(); ++i) {
            nn.hidden[0]->output = std::make_shared<la::gpu::vector_view<double>>(
                la::gpu::vector_view<double>(input_gpu_block.data() + 11 * dim * i, 11 * dim));
    
            autodiff::eval(nn.output, autodiff::gpu::eval_funcs);
    
            la::vector<double> v = la::gpu::to_host(
                autodiff::get_output<la::gpu::vector<double>>(nn.output));
    
            result.push_back(std::vector<double> { v.data(), v.data() + v.size() });
        }
    
        return result;
    }

}
