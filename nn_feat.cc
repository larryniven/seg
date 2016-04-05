#include "scrf/nn_feat.h"
#include "opt/opt.h"
#include "ebt/ebt.h"
#include <fstream>

namespace scrf {

    namespace nn {

        void iadd(param_t& p1, param_t const& p2)
        {
            assert(p1.weight.size() == p1.bias.size());
            assert(p2.weight.size() == p2.bias.size());
            assert(p1.weight.size() == p2.weight.size());

            for (int i = 0; i < p1.weight.size(); ++i) {
                la::iadd(p1.weight[i], p2.weight[i]);
                la::iadd(p1.bias[i], p2.bias[i]);
            }
        }

        void isub(param_t& p1, param_t const& p2)
        {
            assert(p1.weight.size() == p1.bias.size());
            assert(p2.weight.size() == p2.bias.size());
            assert(p1.weight.size() == p2.weight.size());

            for (int i = 0; i < p1.weight.size(); ++i) {
                la::isub(p1.weight[i], p2.weight[i]);
                la::isub(p1.bias[i], p2.bias[i]);
            }
        }

        void imul(param_t& p, double c)
        {
            assert(p.weight.size() == p.bias.size());

            for (int i = 0; i < p.weight.size(); ++i) {
                la::imul(p.weight[i], c);
                la::imul(p.bias[i], c);
            }
        }

        param_t load_nn_param(std::istream& is)
        {
            param_t result;

            ebt::json::json_parser<std::vector<la::matrix<double>>> mat_parser;
            ebt::json::json_parser<std::vector<la::vector<double>>> vec_parser;

            result.weight = mat_parser.parse(is);
            result.bias = vec_parser.parse(is);

            return result;
        }

        param_t load_nn_param(std::string filename)
        {
            std::ifstream ifs { filename };
            return load_nn_param(ifs);
        }

        void save_param(param_t const& p, std::ostream& os)
        {
            ebt::json::dump(p.weight, os);
            os << std::endl;
            ebt::json::dump(p.bias, os);
            os << std::endl;
        }

        void save_param(param_t const& p, std::string filename)
        {
            std::ofstream ofs { filename };
            save_param(p, ofs);
        }

        void adagrad_update(param_t& param, param_t const& grad,
            param_t& accu_grad_sq, double step_size)
        {
            for (int i = 0; i < param.weight.size(); ++i) {
                opt::adagrad_update(param.weight[i], grad.weight[i],
                    accu_grad_sq.weight[i], step_size);
                opt::adagrad_update(param.bias[i], grad.bias[i],
                    accu_grad_sq.bias[i], step_size);
            }
        }

        nn_t make_nn(param_t const& p, autodiff::computation_graph& g)
        {
            assert(p.weight.size() == p.bias.size());

            nn_t result;

            result.hidden.push_back(g.var());

            for (int i = 0; i < p.weight.size(); ++i) {
                result.weight.push_back(g.var(p.weight));
                result.bias.push_back(g.var(p.bias));

                result.hidden.push_back(autodiff::relu(autodiff::add(
                    autodiff::mul(result.weight[i], result.hidden[i]),
                    result.bias[i])));
            }

            return result;
        }

        param_t copy_grad(nn_t const& nn)
        {
            param_t result;

            for (int i = 0; i < nn.weight.size(); ++i) {
                result.weight.push_back(autodiff::get_grad<la::matrix<double>>(nn.weight[i]));
                result.bias.push_back(autodiff::get_grad<la::vector<double>>(nn.bias[i]));
            }

            return result;
        }

        nn_feature::nn_feature(std::unordered_map<std::string, int> const& label_id)
        {
            for (auto& p: label_id) {
                auto& v = label_vec[p.first];
                v.resize(label_id.size());
                v[p.second] = 1;
            }
        }

        void nn_feature::operator()(
            feat_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            feat_t f_base;
            (*base_feat)(f_base, fst, e);

            auto& v = label_vec.at(fst.output(e));
            std::vector<double>& u = f_base.class_vec.at("");
            u.insert(u.end(), v.begin(), v.end());

            nn.hidden[0]->output = std::make_shared<la::vector<double>>(
                la::vector<double>(u));

            autodiff::eval(nn.hidden.back(), autodiff::eval_funcs);

            auto& f_nn = autodiff::get_output<la::vector<double>>(nn.hidden.back());

            auto& o = feat.class_vec[""];

            std::copy(f_nn.data(), f_nn.data() + f_nn.size(), o.begin());
        }

        param_t nn_feature::grad(
            scrf::param_t const& param,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            auto& g = param.class_vec.at("");

            feat_t f;
            (*this)(f, fst, e);

            nn.hidden.back()->grad = std::make_shared<la::vector<double>>(g);

            autodiff::grad(nn.hidden.back(), autodiff::grad_funcs);

            return copy_grad(nn);
        }

        param_t hinge_grad(fst::path<scrf::scrf_t> const& gold_path,
            fst::path<scrf::scrf_t> const& graph_path,
            nn_feature const& nn_feat,
            scrf::param_t const& param)
        {
            param_t result;

            for (auto& e: gold_path.edges()) {
                isub(result, nn_feat.grad(param, *gold_path.data->base_fst->fst, e));
            }

            for (auto& e: graph_path.edges()) {
                iadd(result, nn_feat.grad(param, *graph_path.data->base_fst->fst, e));
            }

            return result;
        }

    }

}
