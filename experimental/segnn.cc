#include "scrf/experimental/segnn.h"
#include "scrf/experimental/iscrf.h"

namespace segnn {

    segnn_feat::segnn_feat(
        scrf::feat_dim_alloc& alloc,
        std::vector<std::vector<double>> const& frames,
        std::shared_ptr<segfeat::feature> const& base_feat,
        residual::nn_param_t const& param, residual::nn_t const& nn)
        : alloc(alloc), frames(frames), base_feat(base_feat), param(param), nn(nn)
    {
        dim = alloc.alloc(1, param.layer.back().weight2.rows());
        residual::resize_as(zero, param);
    }

    void segnn_feat::operator()(scrf::dense_vec& feat, ilat::fst const& a,
        int e) const
    {
        std::vector<double> f;
        f.resize(base_feat->dim(frames.front().size()));
        (*base_feat)(f.data(), frames, a.time(a.tail(e)), a.time(a.head(e)));

        nn.input->output = std::make_shared<la::vector<double>>(std::move(f));

        double *lex_f = iscrf::ilat_lexicalizer().lex(alloc, 1, feat, a, e);
        nn.layer.back().output->output = std::make_shared<la::weak_vector<double>>(
            la::weak_vector<double>(lex_f, param.layer.back().bias2.size()));

        autodiff::eval(nn.layer.back().output, autodiff::eval_funcs);

        // auto& f_out = autodiff::get_output<la::vector<double>>(nn.layer.back().output);
        // std::copy(f_out.data(), f_out.data() + f_out.size(), lex_f);
    }

    void segnn_feat::grad(scrf::dense_vec const& g, ilat::fst const& a, int e)
    {
        std::vector<double> f;
        f.resize(base_feat->dim(frames.front().size()));
        (*base_feat)(f.data(), frames, a.time(a.tail(e)), a.time(a.head(e)));

        nn.input->output = std::make_shared<la::vector<double>>(std::move(f));
        nn.layer.back().output->output = nullptr;
        autodiff::eval(nn.layer.back().output, autodiff::eval_funcs);

        auto& f_out = autodiff::get_output<la::vector<double>>(nn.layer.back().output);

        double const* lex_g = iscrf::ilat_lexicalizer().const_lex(alloc, 1, g, a, e);

        la::vector<double> lex_g_vec;
        lex_g_vec.resize(f_out.size());

        for (int i = 0; i < lex_g_vec.size(); ++i) {
            lex_g_vec(i) = lex_g[i];
        }

        nn.layer.back().output->grad = std::make_shared<la::vector<double>>(
            lex_g_vec);

        // gradient = zero;
        residual::nn_tie_grad(nn, gradient);
        autodiff::grad(nn.layer.back().output, autodiff::grad_funcs);
        // gradient = residual::copy_nn_grad(nn);
        autodiff::clear_grad(nn.layer.back().output);
    }
}
