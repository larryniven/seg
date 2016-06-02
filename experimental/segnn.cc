#include "scrf/experimental/segnn.h"
#include "scrf/experimental/iscrf.h"
#include "opt/opt.h"
#include <fstream>

namespace segnn {

    param_t load_nn_param(std::istream& is)
    {
        param_t result;
        std::string line;

        ebt::json::json_parser<la::matrix<double>> mat_parser;

        result.feat_weight = mat_parser.parse(is);
        std::getline(is, line);
        result.label_weight = mat_parser.parse(is);
        std::getline(is, line);
        result.duration_weight = mat_parser.parse(is);
        std::getline(is, line);

        result.bias = ebt::json::load<la::vector<double>>(is);
        std::getline(is, line);

        return result;
    }

    param_t load_nn_param(std::string filename)
    {
        std::ifstream ifs { filename };
        return load_nn_param(ifs);
    }

    void save_nn_param(param_t const& param, std::ostream& os)
    {
        ebt::json::dump(param.feat_weight, os);
        os << std::endl;
        ebt::json::dump(param.label_weight, os);
        os << std::endl;
        ebt::json::dump(param.duration_weight, os);
        os << std::endl;

        ebt::json::dump(param.bias, os);
        os << std::endl;
    }

    void save_nn_param(param_t const& param, std::string filename)
    {
        std::ofstream ofs { filename };
        save_nn_param(param, ofs);
    }

    void resize_as(param_t& p1, param_t const& p2)
    {
        p1.feat_weight.resize(p2.feat_weight.rows(), p2.feat_weight.cols());
        p1.label_weight.resize(p2.label_weight.rows(), p2.label_weight.cols());
        p1.duration_weight.resize(p2.duration_weight.rows(), p2.duration_weight.cols());

        p1.bias.resize(p2.bias.size());
    }

    void iadd(param_t& p1, param_t const& p2)
    {
        la::iadd(p1.feat_weight, p2.feat_weight);
        // la::iadd(p1.label_weight, p2.label_weight);
        // la::iadd(p1.duration_weight, p2.duration_weight);

        la::iadd(p1.bias, p2.bias);
    }

    void rmsprop_update(param_t& param, param_t const& grad,
        param_t& opt_data, double decay, double step_size)
    {
        opt::rmsprop_update(param.feat_weight, grad.feat_weight,
            opt_data.feat_weight, decay, step_size);
        // opt::rmsprop_update(param.label_weight, grad.label_weight,
        //     opt_data.label_weight, decay, step_size);
        // opt::rmsprop_update(param.duration_weight, grad.duration_weight,
        //     opt_data.duration_weight, decay, step_size);

        opt::rmsprop_update(param.bias, grad.bias,
            opt_data.bias, decay, step_size);
    }

    void adagrad_update(param_t& param, param_t const& grad,
        param_t& accu_grad_sq, double step_size)
    {
        opt::adagrad_update(param.feat_weight, grad.feat_weight,
            accu_grad_sq.feat_weight, step_size);
        // opt::adagrad_update(param.label_weight, grad.label_weight,
        //     accu_grad_sq.label_weight, step_size);
        // opt::adagrad_update(param.duration_weight, grad.duration_weight,
        //     accu_grad_sq.duration_weight, step_size);

        opt::adagrad_update(param.bias, grad.bias,
            accu_grad_sq.bias, step_size);
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

        result.output = autodiff::relu(autodiff::add(std::vector<std::shared_ptr<autodiff::op_t>> {
            autodiff::mul(result.feat_weight, result.feat_embedding),
            // autodiff::mul(result.label_weight, result.label_embedding),
            // autodiff::mul(result.duration_weight, result.duration_embedding),
            result.bias
        }));

        return result;
    }

    param_t copy_nn_grad(nn_t const& nn)
    {
        param_t result;

        result.feat_weight = autodiff::get_grad<la::matrix<double>>(nn.feat_weight);
        // result.label_weight = autodiff::get_grad<la::matrix<double>>(nn.label_weight);
        // result.duration_weight = autodiff::get_grad<la::matrix<double>>(nn.duration_weight);
        result.bias = autodiff::get_grad<la::vector<double>>(nn.bias);

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

        nn.feat_embedding->output = std::make_shared<la::vector<double>>(f);

        // other embedding

        feat.class_vec.resize(1);
        feat.class_vec[0].resize(param.bias.size());
        nn.output->output = std::make_shared<la::weak_vector<double>>(
            la::weak_vector<double>(feat.class_vec[0]));

        autodiff::eval(nn.output, autodiff::eval_funcs);
    }

    void segnn_feat::grad(scrf::dense_vec const& g, ilat::fst const& a, int e)
    {
        la::vector<double> f;
        f.resize(param.feat_weight.cols());
        (*base_feat)(f.data(), frames, a.time(a.tail(e)), a.time(a.head(e)));

        nn.feat_embedding->output = std::make_shared<la::vector<double>>(f);

        // other embedding

        if (g.class_vec.size() != 0 && g.class_vec[0].size() != 0) {
            nn.output->grad = std::make_shared<la::weak_vector<double>>(la::weak_vector<double>(const_cast<scrf::dense_vec&>(g).class_vec[0]));
            autodiff::grad(nn.output, autodiff::grad_funcs);
            gradient = copy_nn_grad(nn);
            autodiff::clear_grad(nn.output);
        }

    }

}
