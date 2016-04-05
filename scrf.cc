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
            result.class_vec[p.first] = std::vector<double>(p.second.data(),
                p.second.data() + p.second.size());
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

    namespace first_order {

        param_t load_param(std::istream& is)
        {
            param_t result;
            std::string line;

            result = param_t { ebt::json::json_parser<
                std::vector<la::vector<double>>>().parse(is) };
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
            os << param.class_vec << std::endl;
        }

        void save_param(param_t const& param, std::string filename)
        {
            std::ofstream ofs { filename };
            save_param(param, ofs);
        }

        void iadd(param_t& p1, param_t const& p2)
        {
            if (p1.class_vec.size() == 0) {
                p1.class_vec.resize(p2.class_vec.size());
            }

            for (int i = 0; i < p2.class_vec.size(); ++i) {
                auto& v = p1.class_vec[i];
                auto& u = p2.class_vec[i];

                if (u.size() == 0) {
                    continue;
                }

                if (v.size() == 0) { 
                    v.resize(u.size());
                } else {
                    assert(v.size() == u.size());
                }

                la::iadd(v, u);
            }
        }

        void isub(param_t& p1, param_t const& p2)
        {
            if (p1.class_vec.size() == 0) {
                p1.class_vec.resize(p2.class_vec.size());
            }

            for (int i = 0; i < p2.class_vec.size(); ++i) {
                auto& v = p1.class_vec[i];
                auto& u = p2.class_vec[i];

                if (u.size() == 0) {
                    continue;
                }

                if (v.size() == 0) { 
                    v.resize(u.size());
                } else {
                    assert(v.size() == u.size());
                }

                la::isub(v, u);
            }
        }

        param_t& operator+=(param_t& p1, param_t const& p2)
        {
            iadd(p1, p2);

            return p1;
        }

        param_t& operator-=(param_t& p1, param_t const& p2)
        {
            isub(p1, p2);

            return p1;
        }

        void imul(param_t& p, double c)
        {
            if (c == 0) {
                p.class_vec.clear();
            }

            for (int i = 0; i < p.class_vec.size(); ++i) {
                if (p.class_vec[i].size() == 0) {
                    continue;
                }

                la::imul(p.class_vec[i], c);
            }
        }

        param_t& operator*=(param_t& p, double c)
        {
            imul(p, c);
            return p;
        }

        real norm(param_t const& p)
        {
            real sum = 0;

            for (int i = 0; i < p.class_vec.size(); ++i) {
                if (p.class_vec[i].size() == 0) {
                    continue;
                }

                double n = la::norm(p.class_vec[i]);
                sum += n * n;
            }

            return std::sqrt(sum);
        }

        real dot(param_t const& p1, param_t const& p2)
        {
            if (p1.class_vec.size() == 0) {
                return 0;
            }

            real sum = 0;

            for (int i = 0; i < p2.class_vec.size(); ++i) {
                if (p1.class_vec[i].size() == 0 || p2.class_vec[i].size() == 0) {
                    continue;
                }

                sum += la::dot(p1.class_vec[i], p2.class_vec[i]);
            }

            return sum;
        }

        void const_step_update_momentum(param_t& theta, param_t const& grad,
            param_t& update, double momentum, double step_size)
        {
            if (update.class_vec.size() == 0) {
                update.class_vec.resize(grad.class_vec.size());
            }

            if (theta.class_vec.size() == 0) {
                theta.class_vec.resize(grad.class_vec.size());
            }

            for (int i = 0; i < grad.class_vec.size(); ++i) {
                if (grad.class_vec[i].size() == 0) {
                    continue;
                }

                theta.class_vec[i].resize(grad.class_vec[i].size());
                update.class_vec[i].resize(grad.class_vec[i].size());

                opt::const_step_update_momentum(theta.class_vec[i], grad.class_vec[i],
                    update.class_vec[i], momentum, step_size);
            }
        }

        void adagrad_update(param_t& param, param_t const& grad,
            param_t& accu_grad_sq, double step_size)
        {
            if (accu_grad_sq.class_vec.size() == 0) {
                accu_grad_sq.class_vec.resize(grad.class_vec.size());
            }

            if (param.class_vec.size() == 0) {
                param.class_vec.resize(grad.class_vec.size());
            }

            for (int i = 0; i < grad.class_vec.size(); ++i) {
                if (grad.class_vec[i].size() == 0) {
                    continue;
                }

                param.class_vec[i].resize(grad.class_vec[i].size());
                accu_grad_sq.class_vec[i].resize(grad.class_vec[i].size());

                opt::adagrad_update(param.class_vec[i], grad.class_vec[i],
                    accu_grad_sq.class_vec[i], step_size);
            }
        }

        scrf_feature::~scrf_feature()
        {}

        feat_dim_alloc::feat_dim_alloc(std::vector<int> const& labels)
            : labels(labels)
        {}

        int feat_dim_alloc::alloc(int order, int dim)
        {
            if (order >= order_dim.size()) {
                order_dim.resize(order + 1);
            }

            int result = order_dim[order];
            order_dim[order] += dim;

            return result;
        }

        scrf_weight::~scrf_weight()
        {}

        std::vector<int> scrf_t::vertices() const
        {
            return fst->vertices();
        }

        std::vector<int> scrf_t::edges() const
        {
            return fst->edges();
        }

        int scrf_t::head(int e) const
        {
            return fst->head(e);
        }

        int scrf_t::tail(int e) const
        {
            return fst->tail(e);
        }

        std::vector<int> scrf_t::in_edges(int v) const
        {
            return fst->in_edges(v);
        }

        std::vector<int> scrf_t::out_edges(int v) const
        {
            return fst->out_edges(v);
        }

        double scrf_t::weight(int e) const
        {
            return (*weight_func)(*fst, e);
        }

        int scrf_t::input(int e) const
        {
            return fst->input(e);
        }

        int scrf_t::output(int e) const
        {
            return fst->output(e);
        }

        std::vector<int> scrf_t::initials() const
        {
            return fst->initials();
        }

        std::vector<int> scrf_t::finals() const
        {
            return fst->finals();
        }

        void scrf_t::feature(param_t& f, int e) const
        {
            (*feature_func)(f, *fst, e);
        }

        double scrf_t::cost(int e) const
        {
            return (*cost_func)(*fst, e);
        }

        fst::path<scrf_t> shortest_path(scrf_t const& s,
            std::vector<int> const& order)
        {
            fst::one_best<scrf_t> best;

            for (auto v: s.initials()) {
                best.extra[v] = {-1, 0};
            }

            best.merge(s, order);

            return best.best_path(s);
        }

        loss_func::~loss_func()
        {}

    }

}
