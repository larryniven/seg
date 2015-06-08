#include "scrf/weiran.h"
#include <fstream>
#include "ebt/ebt.h"
#include "speech/speech.h"
#include "la/la.h"
#include "opt/opt.h"

namespace weiran {

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
            opt::adagrad_update(param.weight[i], grad.weight[i], accu_grad_sq.weight[i], step_size);
        }
    }

    nn_t make_nn(param_t const& param)
    {
        nn_t nn;
        std::vector<real> v;

        auto h = autodiff::var();
        v.resize(param.weight[0].front().size(), 0);
        h->output = std::make_shared<std::vector<real>>(std::move(v));
        nn.layers.push_back(h);

        auto f = autodiff::var();
        v = std::vector<real>();
        v.resize(param.weight[0].size(), 0);
        f->output = std::make_shared<std::vector<real>>(std::move(v));

        auto w1 = autodiff::var(param.weight[0]);
        nn.weights.push_back(w1);
        h = autodiff::relu(autodiff::add(autodiff::mult(w1, h), f));
        v = std::vector<real>();
        v.resize(param.weight[0].size(), 0);
        v.push_back(1);
        h->output = std::make_shared<std::vector<real>>(std::move(v));
        nn.layers.push_back(h);

        // std::cout << param.weight[0].size() << " " << param.weight[0].front().size() << std::endl;
        // std::cout << param.weight.back().size() << " " << param.weight.back().front().size() << std::endl;

        auto w2 = autodiff::var(param.weight.back());
        nn.weights.push_back(w2);
        h = autodiff::logsoftmax(autodiff::mult(w2, h));
        v = std::vector<real>();
        v.resize(param.weight.back().size(), 0);
        h->output = std::make_shared<std::vector<real>>(std::move(v));
        nn.layers.push_back(h);

        /*
        std::cout << "make_nn" << std::endl;
        for (int i = 0; i < nn.layers.size(); ++i) {
            std::cout << i << " " << autodiff::get_output<std::vector<real>>(nn.layers.at(i)).size() << std::endl;
        }
        */

        return nn;
    }

    void move_in_param(nn_t& nn, param_t& param)
    {
        for (int i = 0; i < param.weight.size(); ++i) {
            auto w = nn.layers[i+1]->children[0]->children[0];
            w->output = std::make_shared<std::vector<std::vector<real>>>(std::move(param.weight[i]));
        }
    }

    void move_out_param(nn_t& nn, param_t& param)
    {
        for (int i = 0; i < param.weight.size(); ++i) {
            auto w = nn.layers[i+1]->children[0]->children[0];
            param.weight[i] = std::move(autodiff::get_output<std::vector<std::vector<real>>>(w));
        }
    }

    weiran_feature::weiran_feature(
        std::vector<std::vector<real>> const& frames,
        std::vector<real> const& cm_mean,
        std::vector<real> const& cm_stddev,
        nn_t nn,
        int start_dim,
        int end_dim)
        : frames(frames), cm_mean(cm_mean), cm_stddev(cm_stddev), nn(nn), start_dim(start_dim), end_dim(end_dim)
    {
        if (this->start_dim == -1) {
            this->start_dim = 0;
        }
        if (this->end_dim == -1) {
            this->end_dim = frames.front().size() - 1;
        }
    }

    int weiran_feature::size() const
    {
        return autodiff::get_output<std::vector<real>>(nn.layers.back()).size();
    }

    void weiran_feature::operator()(
        scrf::param_t& feat,
        fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        if (!ebt::in(std::get<0>(e), feat_cache)) {
            int tail = std::get<0>(fst.tail(e));
            int head = std::get<0>(fst.head(e));

            int tail_time = std::min<int>(frames.size() - 1, std::max<int>(0, fst.fst1->data->vertices.at(tail).time));
            int head_time = std::min<int>(frames.size() - 1, std::max<int>(0, fst.fst1->data->vertices.at(head).time));

            std::vector<real> cm_feat = speech::clarkson_moreno_feature(frames, tail_time,
                head_time, start_dim, start_dim + 38);

            for (int i = 0; i < cm_mean.size(); ++i) {
                cm_feat[i] = (cm_feat[i] - cm_mean[i]) / cm_stddev[i];
            }

            nn.layers.at(0)->output = std::make_shared<std::vector<real>>(std::move(cm_feat));

            auto& f = frames.at((tail_time + head_time) / 2);
            std::vector<real> cnn_mult_cache { f.begin() + 39 + start_dim, f.begin() + end_dim + 1 };
            nn.layers.at(1)->children.at(0)->children.at(1)->output
                = std::make_shared<std::vector<real>>(std::move(cnn_mult_cache));

            autodiff::eval(nn.layers.back(), autodiff::eval_funcs);

            feat_cache[std::get<0>(e)] = autodiff::get_output<std::vector<real>>(nn.layers.back());
        }

        auto& u = feat.class_param["[lattice] " + fst.output(e)];
        auto& v = feat_cache.at(std::get<0>(e));

        u.insert(u.end(), v.begin(), v.end());
    }

    param_t hinge_nn_grad(
        nn_t& nn,
        param_t const& nn_param,
        scrf::param_t const& scrf_param,
        fst::path<scrf::scrf_t> const& gold,
        fst::path<scrf::scrf_t> const& cost_aug,
        std::vector<std::string> const& features,
        scrf::composite_feature const& feat_func)
    {
        int index = -1;
        for (int i = 0; i < features.size(); ++i) {
            if (ebt::startswith(features.at(i), std::string("weiran"))) {
                index = i;
                break;
            }
        }

        int start_dim = 0;
        for (int i = 0; i < index; ++i) {
            start_dim += feat_func.features.at(i)->size();
        }
        int end_dim = start_dim + feat_func.features.at(index)->size() - 1;

        param_t nn_grad;

        nn_grad.weight.resize(nn_param.weight.size());
        for (int i = 0; i < nn_param.weight.size(); ++i) {
            nn_grad.weight[i].resize(nn_param.weight[i].size());
            for (int j = 0; j < nn_param.weight[i].size(); ++j) {
                nn_grad.weight[i][j].resize(nn_param.weight[i][j].size());
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
            (*feat_func.features.at(index))(feat, *gold.data->base_fst->fst, e);

            nn.layers.back()->grad = std::make_shared<std::vector<real>>(std::move(top_grad));

            for (auto& w: nn.weights) {
                w->grad = nullptr;
            }

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
            (*feat_func.features.at(index))(feat, *cost_aug.data->base_fst->fst, e);

            nn.layers.back()->grad = std::make_shared<std::vector<real>>(std::move(top_grad));

            for (auto& w: nn.weights) {
                w->grad = nullptr;
            }

            autodiff::grad(nn.layers.back(), autodiff::grad_funcs);

            for (int i = 0; i < nn.weights.size(); ++i) {
                la::iadd(nn_grad.weight[i],
                    autodiff::get_grad<std::vector<std::vector<real>>>(nn.weights.at(i)));
            }
        }

        return nn_grad;
    }

}
