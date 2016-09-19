#include "seg/segnn.h"
#include "seg/iscrf.h"
#include "opt/opt.h"
#include <fstream>

namespace segnn {

    param_t load_nn_param(std::istream& is)
    {
        param_t result;
        std::string line;

        ebt::json::json_parser<la::matrix<double>> mat_parser;

        result.label_embedding = mat_parser.parse(is);
        std::getline(is, line);
        result.duration_embedding = mat_parser.parse(is);
        std::getline(is, line);

        result.feat_weight = mat_parser.parse(is);
        std::getline(is, line);
        result.label_weight = mat_parser.parse(is);
        std::getline(is, line);
        result.duration_weight = mat_parser.parse(is);
        std::getline(is, line);

        result.bias = ebt::json::load<la::vector<double>>(is);
        std::getline(is, line);

        std::getline(is, line);
        int layer = std::stoi(line);

        for (int i = 0; i < layer; ++i) {
            result.layer_weight.push_back(mat_parser.parse(is));
            std::getline(is, line);

            result.layer_bias.push_back(ebt::json::load<la::vector<double>>(is));
            std::getline(is, line);
        }

        return result;
    }

    param_t load_nn_param(std::string filename)
    {
        std::ifstream ifs { filename };
        return load_nn_param(ifs);
    }

    void save_nn_param(param_t const& param, std::ostream& os)
    {
        ebt::json::dump(param.label_embedding, os);
        os << std::endl;
        ebt::json::dump(param.duration_embedding, os);
        os << std::endl;

        ebt::json::dump(param.feat_weight, os);
        os << std::endl;
        ebt::json::dump(param.label_weight, os);
        os << std::endl;
        ebt::json::dump(param.duration_weight, os);
        os << std::endl;

        ebt::json::dump(param.bias, os);
        os << std::endl;

        os << param.layer_weight.size() << std::endl;

        for (int i = 0; i < param.layer_weight.size(); ++i) {
            ebt::json::dump(param.layer_weight[i], os);
            os << std::endl;

            ebt::json::dump(param.layer_bias[i], os);
            os << std::endl;
        }
    }

    void save_nn_param(param_t const& param, std::string filename)
    {
        std::ofstream ofs { filename };
        save_nn_param(param, ofs);
    }

    void resize_as(param_t& p1, param_t const& p2)
    {
        p1.label_embedding.resize(p2.label_embedding.rows(), p2.label_embedding.cols());
        p1.duration_embedding.resize(p2.duration_embedding.rows(), p2.duration_embedding.cols());

        p1.feat_weight.resize(p2.feat_weight.rows(), p2.feat_weight.cols());
        p1.label_weight.resize(p2.label_weight.rows(), p2.label_weight.cols());
        p1.duration_weight.resize(p2.duration_weight.rows(), p2.duration_weight.cols());

        p1.bias.resize(p2.bias.size());

        p1.layer_weight.resize(p2.layer_weight.size());
        p1.layer_bias.resize(p2.layer_bias.size());

        for (int i = 0; i < p1.layer_weight.size(); ++i) {
            p1.layer_weight[i].resize(p2.layer_weight[i].rows(), p2.layer_weight[i].cols());
            p1.layer_bias[i].resize(p2.layer_bias[i].size());
        }
    }

    void iadd(param_t& p1, param_t const& p2)
    {
        la::iadd(p1.label_embedding, p2.label_embedding);
        la::iadd(p1.duration_embedding, p2.duration_embedding);

        la::iadd(p1.feat_weight, p2.feat_weight);
        la::iadd(p1.label_weight, p2.label_weight);
        la::iadd(p1.duration_weight, p2.duration_weight);

        la::iadd(p1.bias, p2.bias);

        for (int i = 0; i < p1.layer_weight.size(); ++i) {
            la::iadd(p1.layer_weight[i], p2.layer_weight[i]);
            la::iadd(p1.layer_bias[i], p2.layer_bias[i]);
        }
    }

    void rmsprop_update(param_t& param, param_t const& grad,
        param_t& opt_data, double decay, double step_size)
    {
        opt::rmsprop_update(param.label_embedding, grad.label_embedding,
            opt_data.label_embedding, decay, step_size);
        opt::rmsprop_update(param.duration_embedding, grad.duration_embedding,
            opt_data.duration_embedding, decay, step_size);

        opt::rmsprop_update(param.feat_weight, grad.feat_weight,
            opt_data.feat_weight, decay, step_size);
        opt::rmsprop_update(param.label_weight, grad.label_weight,
            opt_data.label_weight, decay, step_size);
        opt::rmsprop_update(param.duration_weight, grad.duration_weight,
            opt_data.duration_weight, decay, step_size);

        opt::rmsprop_update(param.bias, grad.bias,
            opt_data.bias, decay, step_size);

        for (int i = 0; i < param.layer_weight.size(); ++i) {
            opt::rmsprop_update(param.layer_weight[i], grad.layer_weight[i],
                opt_data.layer_weight[i], decay, step_size);
            opt::rmsprop_update(param.layer_bias[i], grad.layer_bias[i],
                opt_data.layer_bias[i], decay, step_size);
        }
    }

    void adagrad_update(param_t& param, param_t const& grad,
        param_t& accu_grad_sq, double step_size)
    {
        opt::adagrad_update(param.label_embedding, grad.label_embedding,
            accu_grad_sq.label_embedding, step_size);
        opt::adagrad_update(param.duration_embedding, grad.duration_embedding,
            accu_grad_sq.duration_embedding, step_size);

        opt::adagrad_update(param.feat_weight, grad.feat_weight,
            accu_grad_sq.feat_weight, step_size);
        opt::adagrad_update(param.label_weight, grad.label_weight,
            accu_grad_sq.label_weight, step_size);
        opt::adagrad_update(param.duration_weight, grad.duration_weight,
            accu_grad_sq.duration_weight, step_size);

        opt::adagrad_update(param.bias, grad.bias,
            accu_grad_sq.bias, step_size);

        for (int i = 0; i < param.layer_weight.size(); ++i) {
            opt::adagrad_update(param.layer_weight[i], grad.layer_weight[i],
                accu_grad_sq.layer_weight[i], step_size);
            opt::adagrad_update(param.layer_bias[i], grad.layer_bias[i],
                accu_grad_sq.layer_bias[i], step_size);
        }
    }

    nn_t make_nn(autodiff::computation_graph& graph,
        param_t const& param)
    {
        nn_t result;

        result.feat_weight = graph.var(param.feat_weight);
        result.label_weight = graph.var(param.label_weight);
        result.duration_weight = graph.var(param.duration_weight);
        result.bias = graph.var(param.bias);

        result.feat_embedding = graph.var();
        result.label_embedding = graph.var();
        result.duration_embedding = graph.var();

        result.layer.push_back(autodiff::relu(
            autodiff::add(std::vector<std::shared_ptr<autodiff::op_t>> {
                autodiff::mul(result.feat_weight, result.feat_embedding),
                autodiff::mul(result.label_weight, result.label_embedding),
                autodiff::mul(result.duration_weight, result.duration_embedding),
                result.bias
            })));

        for (int i = 0; i < param.layer_weight.size(); ++i) {
            result.layer_weight.push_back(graph.var(param.layer_weight[i]));
	    result.layer_bias.push_back(graph.var(param.layer_bias[i]));

            result.layer.push_back(autodiff::relu(
                autodiff::add(
                    autodiff::mul(result.layer_weight[i], result.layer[i]),
                    result.layer_bias[i]
                )));
        }

        result.output = result.layer.back();

        return result;
    }

    param_t copy_nn_grad(nn_t const& nn)
    {
        param_t result;

        result.feat_weight = autodiff::get_grad<la::matrix<double>>(nn.feat_weight);
        result.label_weight = autodiff::get_grad<la::matrix<double>>(nn.label_weight);
        result.duration_weight = autodiff::get_grad<la::matrix<double>>(nn.duration_weight);
        result.bias = autodiff::get_grad<la::vector<double>>(nn.bias);

        for (int i = 0; i < nn.layer_weight.size(); ++i) {
            result.layer_weight.push_back(autodiff::get_grad<la::matrix<double>>(nn.layer_weight[i]));
            result.layer_bias.push_back(autodiff::get_grad<la::vector<double>>(nn.layer_bias[i]));
        }

        return result;
    }

    segnn_feat::segnn_feat(
        std::vector<std::vector<double>> const& frames,
        std::shared_ptr<segfeat::feature> const& base_feat,
        param_t const& param, nn_t const& nn)
        : frames(frames), base_feat(base_feat), param(param), nn(nn)
    {
    }

    void segnn_feat::operator()(scrf::dense_vec& feat, ilat::fst const& a,
        int e) const
    {
        la::vector<double> f;
        f.resize(param.feat_weight.cols());
        (*base_feat)(f.data(), frames, a.time(a.tail(e)), a.time(a.head(e)));

        nn.feat_embedding->output = std::make_shared<la::weak_vector<double>>(
            la::weak_vector<double>(f));
        nn.label_embedding->output = std::make_shared<la::weak_vector<double>>(
            la::weak_vector<double>(const_cast<double*>(param.label_embedding.data())
                + param.label_embedding.cols() * (a.output(e) - 1),
                param.label_embedding.cols()));

        int duration = a.time(a.head(e)) - a.time(a.tail(e));

        nn.duration_embedding->output = std::make_shared<la::weak_vector<double>>(
            la::weak_vector<double>(const_cast<double*>(param.duration_embedding.data())
                + param.duration_embedding.cols() * (duration - 1),
                param.duration_embedding.cols()));

        feat.class_vec.resize(1);
        feat.class_vec[0].resize(param.bias.size());
        nn.output->output = std::make_shared<la::weak_vector<double>>(
            la::weak_vector<double>(feat.class_vec[0]));

        autodiff::eval(nn.output, autodiff::eval_funcs);

    }

    void segnn_feat::grad(scrf::dense_vec const& g, ilat::fst const& a, int e)
    {
        if (g.class_vec.size() != 0 && g.class_vec[0].size() != 0) {
            la::vector<double> f;
            f.resize(param.feat_weight.cols());
            (*base_feat)(f.data(), frames, a.time(a.tail(e)), a.time(a.head(e)));

            nn.feat_embedding->output = std::make_shared<la::vector<double>>(f);
            nn.label_embedding->output = std::make_shared<la::weak_vector<double>>(
                la::weak_vector<double>(const_cast<double*>(param.label_embedding.data())
                    + param.label_embedding.cols() * (a.output(e) - 1),
                    param.label_embedding.cols()));
            nn.output->output = nullptr;

            int duration = a.time(a.head(e)) - a.time(a.tail(e));

            nn.duration_embedding->output = std::make_shared<la::weak_vector<double>>(
                la::weak_vector<double>(const_cast<double*>(param.duration_embedding.data())
                    + param.duration_embedding.cols() * (duration - 1),
                    param.duration_embedding.cols()));

            autodiff::eval(nn.output, autodiff::eval_funcs);

            nn.output->grad = std::make_shared<la::weak_vector<double>>(
                la::weak_vector<double>(const_cast<scrf::dense_vec&>(g).class_vec[0]));
            autodiff::grad(nn.output, autodiff::grad_funcs);
            gradient = copy_nn_grad(nn);

            gradient.label_embedding.resize(param.label_embedding.rows(), param.label_embedding.cols());
            auto& label_embedding_grad = autodiff::get_grad<la::vector<double>>(nn.label_embedding);
            for (int i = 0; i < param.label_embedding.cols(); ++i) {
                gradient.label_embedding(a.output(e) - 1, i) += label_embedding_grad(i);
            }

            gradient.duration_embedding.resize(param.duration_embedding.rows(), param.duration_embedding.cols());
            auto& duration_embedding_grad = autodiff::get_grad<la::vector<double>>(nn.duration_embedding);
            for (int i = 0; i < param.duration_embedding.cols(); ++i) {
                gradient.duration_embedding(duration - 1, i) += duration_embedding_grad(i);
            }

            autodiff::clear_grad(nn.output);
        } else {
            resize_as(gradient, param);
        }

    }

}
