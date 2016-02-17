#include "scrf/scrf.h"
#include <istream>
#include <fstream>
#include <cassert>
#include "ebt/ebt.h"
#include "opt/opt.h"
#include "la/la.h"

namespace scrf {

    feat_t to_feat(param_t param)
    {
        feat_t result;

        for (auto& p: param.class_vec) {
            result.class_vec[p.first] = std::vector<double>(p.second.data(), p.second.data() + p.second.size());
        }

        return result;
    }

    param_t to_param(feat_t f)
    {
        param_t result;

        for (auto& p: f.class_vec) {
            result.class_vec[p.first] = la::vector<double>(std::move(p.second));
        }

        return result;
    }

    param_t load_param(std::istream& is)
    {
        param_t result;
        std::string line;

        result = to_param(feat_t { ebt::json::json_parser<
            std::unordered_map<std::string, std::vector<real>>>().parse(is) });
        std::getline(is, line);

        return result;
    }

    param_t load_param(std::string filename)
    {
        std::ifstream ifs { filename };
        return load_param(ifs);
    }

    void save_param(param_t const& param, std::ostream& os)
    {
        os << to_feat(param).class_vec << std::endl;
    }

    void save_param(param_t const& param, std::string filename)
    {
        std::ofstream ofs { filename };
        save_param(param, ofs);
    }

    void iadd(param_t& p1, param_t const& p2)
    {
        for (auto& p: p2.class_vec) {
            auto& v = p1.class_vec[p.first];

            v.resize(std::max(v.size(), p.second.size()));

            la::iadd(v, p.second);
        }
    }

    void isub(param_t& p1, param_t const& p2)
    {
        for (auto& p: p2.class_vec) {
            auto& v = p1.class_vec[p.first];

            v.resize(std::max(v.size(), p.second.size()));

            la::isub(v, p.second);
        }
    }

    param_t& operator-=(param_t& p1, param_t const& p2)
    {
        for (auto& p: p2.class_vec) {
            auto& v = p1.class_vec[p.first];

            v.resize(std::max(v.size(), p.second.size()));

            la::isub(v, p.second);
        }

        return p1;
    }

    param_t& operator+=(param_t& p1, param_t const& p2)
    {
        for (auto& p: p2.class_vec) {
            auto& v = p1.class_vec[p.first];

            v.resize(std::max(v.size(), p.second.size()));

            la::iadd(v, p.second);
        }

        return p1;
    }

    param_t& operator*=(param_t& p, real c)
    {
        if (c == 0) {
            p.class_vec.clear();
        }

        for (auto& t: p.class_vec) {
            la::imul(t.second, c);
        }

        return p;
    }

    real norm(param_t const& p)
    {
        real sum = 0;

        for (auto& t: p.class_vec) {
            double n = la::norm(t.second);
            sum += n * n;
        }

        return std::sqrt(sum);
    }

    real dot(param_t const& p1, param_t const& p2)
    {
        real sum = 0;

        for (auto& p: p2.class_vec) {
            if (!ebt::in(p.first, p1.class_vec)) {
                continue;
            }

            auto& v = p1.class_vec.at(p.first);

            sum += la::dot(v, p.second);
        }

        return sum;
    }

    void const_step_update_momentum(param_t& theta, param_t grad,
        param_t& update, real momentum, real step_size)
    {
        std::unordered_map<std::string, int> classes;

        unsigned int size = 0;

        for (auto& p: grad.class_vec) {
            classes[p.first] = p.second.size();
        }

        for (auto& p: update.class_vec) {
            classes[p.first] = p.second.size();
        }

        for (auto& p: theta.class_vec) {
            classes[p.first] = p.second.size();
        }

        for (auto& p: classes) {
            if (!ebt::in(p.first, theta.class_vec)) {
                theta.class_vec[p.first].resize(p.second);
            }

            if (!ebt::in(p.first, update.class_vec)) {
                update.class_vec[p.first].resize(p.second);
            }

            if (!ebt::in(p.first, grad.class_vec)) {
                grad.class_vec[p.first].resize(p.second);
            }

            opt::const_step_update_momentum(theta.class_vec.at(p.first), grad.class_vec.at(p.first),
                update.class_vec.at(p.first), momentum, step_size);
        }
    }

    void adagrad_update(param_t& param, param_t const& grad,
        param_t& accu_grad_sq, real step_size)
    {
        for (auto& p: grad.class_vec) {
            if (!ebt::in(p.first, param.class_vec)) {
                param.class_vec[p.first].resize(p.second.size());
            }
            if (!ebt::in(p.first, accu_grad_sq.class_vec)) {
                accu_grad_sq.class_vec[p.first].resize(p.second.size());
            }
            opt::adagrad_update(param.class_vec.at(p.first), p.second,
                accu_grad_sq.class_vec.at(p.first), step_size);
        }
    }

    scrf_weight::~scrf_weight()
    {}

    scrf_feature::~scrf_feature()
    {}

    std::vector<std::tuple<int, int>> topo_order(scrf_t const& scrf)
    {
        auto const& lat = *(scrf.fst->fst1);
        auto const& lm = *(scrf.fst->fst2);

        auto lat_order = fst::topo_order(lat);
        auto lm_vertices = lm.vertices();

        std::vector<std::tuple<int, int>> result;

        std::reverse(lm_vertices.begin(), lm_vertices.end());

        for (auto u: lat_order) {
            for (auto v: lm_vertices) {
                result.push_back(std::make_tuple(u, v));
            }
        }

        return result;
    }

    fst::path<scrf_t> shortest_path(scrf_t const& s,
        std::vector<std::tuple<int, int>> const& order)
    {
        fst::one_best<scrf_t> best;

        for (auto v: s.initials()) {
            best.extra[v] = {std::make_tuple(-1, -1), 0};
        }

        best.merge(s, order);

        return best.best_path(s);
    }

    loss_func::~loss_func()
    {}

}
