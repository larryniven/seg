#include "scrf/scrf.h"
#include <istream>
#include <fstream>
#include "ebt/ebt.h"
#include "opt/opt.h"
#include "scrf/weiran.h"
#include "la/la.h"

namespace scrf {

    param_t load_param(std::istream& is)
    {
        param_t result;
        std::string line;

        result.class_param = ebt::json::json_parser<
            std::unordered_map<std::string, std::vector<real>>>().parse(is);
        std::getline(is, line);

        return result;
    }

    param_t load_param(std::string filename)
    {
        std::ifstream ifs { filename };
        return load_param(ifs);
    }

    void save_param(std::ostream& os, param_t const& param)
    {
        os << param.class_param << std::endl;
    }

    void save_param(std::string filename, param_t const& param)
    {
        std::ofstream ofs { filename };
        save_param(ofs, param);
    }

    param_t& operator-=(param_t& p1, param_t const& p2)
    {
        for (auto& p: p2.class_param) {
            auto& v = p1.class_param[p.first];

            v.resize(std::max(v.size(), p.second.size()));

            la::isub(v, p.second);
        }

        return p1;
    }

    param_t& operator+=(param_t& p1, param_t const& p2)
    {
        for (auto& p: p2.class_param) {
            auto& v = p1.class_param[p.first];

            v.resize(std::max(v.size(), p.second.size()));

            la::iadd(v, p.second);
        }

        return p1;
    }

    param_t& operator*=(param_t& p, real c)
    {
        if (c == 0) {
            p.class_param.clear();
        }

        for (auto& t: p.class_param) {
            la::imul(t.second, c);
        }

        return p;
    }

    real dot(param_t const& p1, param_t const& p2)
    {
        real sum = 0;

        for (auto& p: p2.class_param) {
            if (!ebt::in(p.first, p1.class_param)) {
                continue;
            }

            auto& v = p1.class_param.at(p.first);

            sum += la::dot(v, p.second);
        }

        return sum;
    }

    void adagrad_update(param_t& param, param_t const& grad,
        param_t& accu_grad_sq, real step_size)
    {
        for (auto& p: grad.class_param) {
            if (!ebt::in(p.first, param.class_param)) {
                param.class_param[p.first].resize(p.second.size());
            }
            if (!ebt::in(p.first, accu_grad_sq.class_param)) {
                accu_grad_sq.class_param[p.first].resize(p.second.size());
            }
            opt::adagrad_update(param.class_param.at(p.first), p.second,
                accu_grad_sq.class_param.at(p.first), step_size);
        }
    }

    scrf_feature::~scrf_feature()
    {}

    collapsed_feature::collapsed_feature(
        std::string name,
        std::shared_ptr<scrf_feature> feat_func)
        : name_(name), feat_func(feat_func)
    {}

    int collapsed_feature::size() const
    {
        return 0;
    }

    std::string collapsed_feature::name() const
    {
        return "";
    }

    void collapsed_feature::operator()(
        param_t& feat,
        fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        param_t feat_tmp;
        (*feat_func)(feat_tmp, fst, e);

        std::vector<real> result;
        for (auto& p: feat_tmp.class_param) {
            result.insert(result.end(), p.second.begin(), p.second.end());
        }
        feat.class_param[name_] = std::move(result);
    }

    composite_feature::composite_feature(std::string name)
        : name_(name)
    {}

    int composite_feature::size() const
    {
        int sum = 0;
        for (auto& f: features) {
            sum += f->size();
        }
        return sum;
    }

    std::string composite_feature::name() const
    {
        return name_;
    }

    void composite_feature::operator()(
        param_t& feat,
        fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        for (auto& f: features) {
            (*f)(feat, fst, e);
        }
    }
        
    scrf_weight::~scrf_weight()
    {}

    linear_weight::linear_weight(param_t const& param, scrf_feature const& feat_func)
        : param(param), feat_func(feat_func)
    {}

    real linear_weight::operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        param_t feat;

        feat_func(feat, fst, e);

        return dot(param, feat);
    }

    namespace score {

        linear_score::linear_score(param_t const& param,
                std::shared_ptr<scrf_feature> feat)
            : param(param), feat(feat)
        {}

        real linear_score::operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            param_t p;
            (*feat)(p, fst, e);
            real s = dot(param, p);

            return s;
        }

        label_score::label_score(param_t const& param,
                std::shared_ptr<scrf_feature> feat)
            : param(param), feat(feat)
        {}

        real label_score::operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            if (ebt::in(fst.output(e), cache)) {
                return cache[fst.output(e)];
            }

            param_t p;
            (*feat)(p, fst, e);
            real s = dot(param, p);

            cache[fst.output(e)] = s;

            return s;
        }

        lm_score::lm_score(param_t const& param,
                std::shared_ptr<scrf_feature> feat)
            : param(param), feat(feat)
        {}

        real lm_score::operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            if (ebt::in(std::get<1>(e), cache)) {
                return cache[std::get<1>(e)];
            }

            param_t p;
            (*feat)(p, fst, e);
            real s = dot(param, p);

            cache[std::get<1>(e)] = s;

            return s;
        }

        lattice_score::lattice_score(param_t const& param,
                std::shared_ptr<scrf_feature> feat)
            : param(param), feat(feat)
        {}

        real lattice_score::operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            /*
            if (ebt::in(std::make_tuple(std::get<0>(e), fst.output(e)), cache)) {
                return cache[std::make_tuple(std::get<0>(e), fst.output(e))];
            }
            */

            param_t p;
            (*feat)(p, fst, e);
            real s = dot(param, p);

            // cache[std::make_tuple(std::get<0>(e), fst.output(e))] = s;

            return s;
        }

    }

    std::vector<std::tuple<int, int>> topo_order(scrf_t const& scrf)
    {
        auto const& lat = *(scrf.fst->fst1);
        auto const& lm = *(scrf.fst->fst2);

        std::vector<std::tuple<int, int>> result;

        auto lm_vertices = lm.vertices();
        std::reverse(lm_vertices.begin(), lm_vertices.end());

        for (auto u: lat.vertices()) {
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

    lattice::fst load_gold(std::istream& is)
    {
        std::string line;
        std::getline(is, line);
    
        lattice::fst_data result;

        int v = 0;
        result.initials.push_back(v);
        result.vertices.push_back(lattice::vertex_data{});
        result.vertices.at(v).time = 0;
    
        while (std::getline(is, line) && line != ".") {
            std::vector<std::string> parts = ebt::split(line);
    
            long tail_time = long(std::stol(parts.at(0)) / 1e5);
            long head_time = long(std::stol(parts.at(1)) / 1e5);
    
            if (tail_time == head_time) {
                continue;
            }

            int e = result.edges.size();
            int u = result.vertices.size();

            result.vertices.push_back(lattice::vertex_data{});
            result.vertices.at(u).time = head_time;

            if (u > int(result.in_edges.size()) - 1) {
                result.in_edges.resize(u + 1);
                result.in_edges_map.resize(u + 1);
            }

            result.in_edges[u].push_back(e);
            result.in_edges_map[u][parts.at(2)].push_back(e);

            if (v > int(result.out_edges.size()) - 1) {
                result.out_edges.resize(v + 1);
                result.out_edges_map.resize(v + 1);
            }

            result.out_edges[v].push_back(e);
            result.out_edges_map[v][parts.at(2)].push_back(e);

            result.edges.push_back(lattice::edge_data{});
            result.edges.at(e).tail = v;
            result.edges.at(e).head = u;
            result.edges.at(e).label = parts.at(2);

            v = u;
        }

        result.finals.push_back(v);

        lattice::fst f;
        f.data = std::make_shared<lattice::fst_data>(std::move(result));

        return f;
    }
    
    lattice::fst make_segmentation_lattice(int frames, int max_seg)
    {
        lattice::fst_data data;

        data.vertices.resize(frames + 1);
        for (int i = 0; i < frames + 1; ++i) {
            data.vertices.at(i).time = i;
        }

        data.in_edges.resize(frames + 1);
        data.out_edges.resize(frames + 1);
        data.in_edges_map.resize(frames + 1);
        data.out_edges_map.resize(frames + 1);

        for (int i = 0; i < frames + 1; ++i) {
            for (int j = 1; j <= max_seg; ++j) {
                int tail = i;
                int head = i + j;

                if (head > frames) {
                    continue;
                }

                data.edges.push_back(lattice::edge_data {"<label>", tail, head});
                int e = data.edges.size() - 1;

                data.in_edges.at(head).push_back(e);
                data.in_edges_map.at(head)["<label>"].push_back(e);
                data.out_edges.at(tail).push_back(e);
                data.in_edges_map.at(tail)["<label>"].push_back(e);
            }
        }

        data.initials.push_back(0);
        data.finals.push_back(frames);

        lattice::fst f;
        f.data = std::make_shared<lattice::fst_data>(std::move(data));

        return f;
    }

    scrf_t make_gold_scrf(lattice::fst gold_lat,
        std::shared_ptr<lm::fst> lm)
    {
        gold_lat.data = std::make_shared<lattice::fst_data>(*(gold_lat.data));
        lattice::add_eps_loops(gold_lat);
        fst::composed_fst<lattice::fst, lm::fst> gold_lm_lat;
        gold_lm_lat.fst1 = std::make_shared<lattice::fst>(std::move(gold_lat));
        gold_lm_lat.fst2 = lm;

        scrf_t gold;
        gold.fst = std::make_shared<decltype(gold_lm_lat)>(gold_lm_lat);

        return gold;
    }

    loss_func::~loss_func()
    {}

}
