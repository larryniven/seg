#include "scrf/nn.h"
#include <fstream>
#include "ebt/ebt.h"
#include "speech/speech.h"
#include "la/la.h"
#include "opt/opt.h"

namespace nn {

    param_t load_param(std::istream& is)
    {
        param_t param;
        std::string line;

        ebt::json::json_parser<decltype(param.weight)> weight_parser;
        param.weight = weight_parser.parse(is);
        std::getline(is, line);

        return param;
    }

    param_t load_param(std::string filename)
    {
        std::ifstream ifs { filename };

        return load_param(ifs);
    }

    void save_param(param_t const& param, std::ostream& os)
    {
        os << param.weight << std::endl;
    }
    
    void save_param(param_t const& param, std::string filename)
    {
        std::ofstream ofs { filename };
        save_param(param, ofs);
    }

    void adagrad_update(param_t& param, param_t const& grad,
        param_t& accu_grad_sq, double step_size)
    {
        for (int i = 0; i < param.weight.size(); ++i) {
            opt::adagrad_update(param.weight[i], grad.weight[i],
                accu_grad_sq.weight[i], step_size);
        }
    }

    nn_t make_nn(param_t const& param)
    {
        nn_t nn;
        std::vector<real> v;

        auto h = nn.graph.var();
        v.resize(param.weight[0].front().size(), 0);
        h->output = std::make_shared<std::vector<real>>(std::move(v));
        nn.layers.push_back(h);

        for (int i = 0; i < param.weight.size() - 1; ++i) {
            auto w = nn.graph.var(param.weight[i]);
            nn.weights.push_back(w);
            h = autodiff::relu(autodiff::mult(w, h));
            v = std::vector<real>();
            v.resize(param.weight[i].size(), 0);
            v.push_back(1);
            h->output = std::make_shared<std::vector<real>>(std::move(v));
            nn.layers.push_back(h);
        }

        auto w = nn.graph.var(param.weight.back());
        nn.weights.push_back(w);
        h = autodiff::logsoftmax(autodiff::mult(w, h));
        v = std::vector<real>();
        v.resize(param.weight.back().size(), 0);
        h->output = std::make_shared<std::vector<real>>(std::move(v));
        nn.layers.push_back(h);

        return nn;
    }

    void move_in_param(nn_t& nn, param_t& param)
    {
        for (int i = 0; i < nn.weights.size(); ++i) {
            auto w = nn.weights[i];
            w->output = std::make_shared<std::vector<std::vector<real>>>(
                std::move(param.weight[i]));
        }
    }

    void move_out_param(nn_t& nn, param_t& param)
    {
        for (int i = 0; i < nn.weights.size(); ++i) {
            auto w = nn.weights[i];
            param.weight[i] = std::move(
                autodiff::get_output<std::vector<std::vector<real>>>(w));
        }
    }

    nn_feature::nn_feature(
        std::vector<std::vector<real>> const& frames,
        std::vector<real> const& cm_mean,
        std::vector<real> const& cm_stddev,
        nn_t nn,
        int start_dim,
        int end_dim)
        : frames(frames), cm_mean(cm_mean)
        , cm_stddev(cm_stddev), nn(nn)
        , start_dim(start_dim), end_dim(end_dim)
    {
        if (this->start_dim == -1) {
            this->start_dim = 0;
        }
        if (this->end_dim == -1) {
            this->end_dim = frames.front().size() - 1;
        }
    }

    int nn_feature::size() const
    {
        return autodiff::get_output<std::vector<real>>(nn.layers.back()).size();
    }

    std::string nn_feature::name() const
    {
        return "nn";
    }

    void nn_feature::operator()(
        scrf::param_t& feat,
        fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        if (!ebt::in(std::get<0>(e), feat_cache)) {
            int tail = std::get<0>(fst.tail(e));
            int head = std::get<0>(fst.head(e));

            int tail_time = std::min<int>(frames.size() - 1,
                std::max<int>(0, fst.fst1->data->vertices.at(tail).time));
            int head_time = std::min<int>(frames.size() - 1,
                std::max<int>(0, fst.fst1->data->vertices.at(head).time));

            std::vector<real> cm_feat = speech::clarkson_moreno_feature(frames, tail_time,
                head_time, start_dim, start_dim + 38);

            for (int i = 0; i < cm_mean.size(); ++i) {
                cm_feat[i] = (cm_feat[i] - cm_mean[i]) / cm_stddev[i];
            }

            nn.layers.at(0)->output = std::make_shared<std::vector<real>>(std::move(cm_feat));

            autodiff::eval(nn.layers.back(), autodiff::eval_funcs);

            feat_cache[std::get<0>(e)] = autodiff::get_output<std::vector<real>>(
                nn.layers.back());

        }

        auto& u = feat.class_param["[lattice] " + fst.output(e)];
        auto& v = feat_cache.at(std::get<0>(e));

        u.insert(u.end(), v.begin(), v.end());
    }

    param_t hinge_nn_grad(
        nn_t& nn,
        scrf::param_t const& scrf_param,
        fst::path<scrf::scrf_t> const& gold,
        fst::path<scrf::scrf_t> const& cost_aug,
        scrf::composite_feature const& feat_func)
    {
        scrf::composite_feature const& lattice_feat =
            static_cast<scrf::composite_feature const&>(*feat_func.features[0]);

        int index = -1;
        for (int i = 0; i < lattice_feat.features.size(); ++i) {
            if (ebt::startswith(lattice_feat.features.at(i)->name(), std::string("weiran"))) {
                index = i;
                break;
            }
        }

        int start_dim = 0;
        for (int i = 0; i < index; ++i) {
            start_dim += lattice_feat.features.at(i)->size();
        }
        int end_dim = start_dim + lattice_feat.features.at(index)->size() - 1;

        param_t nn_grad;

        nn_grad.weight.resize(nn.weights.size());
        for (int i = 0; i < nn.weights.size(); ++i) {
            auto& w = autodiff::get_output<std::vector<std::vector<real>>>(nn.weights[i]);
            nn_grad.weight[i].resize(w.size());
            for (int j = 0; j < w.size(); ++j) {
                nn_grad.weight[i][j].resize(w[j].size());
            }
        }

        for (auto& e: gold.edges()) {
            std::vector<real> top_grad;
            top_grad.resize(end_dim - start_dim + 1);

            std::string key = "[lattice] " + gold.output(e);

            if (!ebt::in(key, scrf_param.class_param)) {
                continue;
            }

            auto& v = scrf_param.class_param.at("[lattice] " + gold.output(e));

            for (int i = start_dim; i < end_dim + 1; ++i) {
                top_grad[i - start_dim] -= v[i];
            }

            scrf::param_t feat;

            // force fead forward
            (*lattice_feat.features.at(index))(feat, *gold.data->base_fst->fst, e);

            autodiff::clear_grad(nn.layers.back());
            nn.layers.back()->grad = std::make_shared<std::vector<real>>(std::move(top_grad));

            autodiff::grad(nn.layers.back(), autodiff::grad_funcs);

            for (int i = 0; i < nn.weights.size(); ++i) {
                la::iadd(nn_grad.weight[i],
                    autodiff::get_grad<std::vector<std::vector<real>>>(nn.weights.at(i)));
            }
        }

        for (auto& e: cost_aug.edges()) {
            std::vector<real> top_grad;
            top_grad.resize(end_dim - start_dim + 1);

            std::string key = "[lattice] " + cost_aug.output(e);

            if (!ebt::in(key, scrf_param.class_param)) {
                continue;
            }

            auto& v = scrf_param.class_param.at("[lattice] " + cost_aug.output(e));
            for (int i = start_dim; i < end_dim + 1; ++i) {
                top_grad[i - start_dim] += v[i];
            }

            scrf::param_t feat;

            // force fead forward
            (*lattice_feat.features.at(index))(feat, *cost_aug.data->base_fst->fst, e);

            autodiff::clear_grad(nn.layers.back());
            nn.layers.back()->grad = std::make_shared<std::vector<real>>(std::move(top_grad));

            autodiff::grad(nn.layers.back(), autodiff::grad_funcs);

            for (int i = 0; i < nn.weights.size(); ++i) {
                la::iadd(nn_grad.weight[i],
                    autodiff::get_grad<std::vector<std::vector<real>>>(nn.weights.at(i)));
            }
        }

        return nn_grad;
    }

}
