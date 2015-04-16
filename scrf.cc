#include "scrf/scrf.h"
#include <istream>
#include <fstream>
#include "ebt/ebt.h"
#include "opt/opt.h"

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

            for (int i = 0; i < p.second.size(); ++i) {
                v.at(i) -= p.second.at(i);
            }
        }

        return p1;
    }

    param_t& operator+=(param_t& p1, param_t const& p2)
    {
        for (auto& p: p2.class_param) {
            auto& v = p1.class_param[p.first];

            v.resize(std::max(v.size(), p.second.size()));

            for (int i = 0; i < p.second.size(); ++i) {
                v.at(i) += p.second.at(i);
            }
        }

        return p1;
    }

    real dot(param_t const& p1, param_t const& p2)
    {
        real sum = 0;

        for (auto& p: p2.class_param) {
            if (!ebt::in(p.first, p1.class_param)) {
                continue;
            }

            auto& v = p1.class_param.at(p.first);

            for (int i = 0; i < p.second.size(); ++i) {
                sum += v.at(i) * p.second.at(i);
            }
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

    real composite_weight::operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        real sum = 0;

        for (auto& w: weights) {
            sum += (*w)(fst, e);
        }

        return sum;
    }

    linear_score::linear_score(param_t const& param, scrf_feature const& feat_func)
        : param(param), feat_func(feat_func)
    {}

    real linear_score::operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        param_t feat;

        feat_func(feat, fst, e);

        return dot(param, feat);
    }

    std::vector<scrf_t::vertex_type> scrf_t::vertices() const
    {
        return fst->vertices();
    }

    std::vector<scrf_t::edge_type> scrf_t::edges() const
    {
        return fst->edges();
    }

    scrf_t::vertex_type scrf_t::head(scrf_t::edge_type const& e) const
    {
        return fst->head(e);
    }

    scrf_t::vertex_type scrf_t::tail(scrf_t::edge_type const& e) const
    {
        return fst->tail(e);
    }

    std::vector<scrf_t::edge_type> scrf_t::in_edges(scrf_t::vertex_type const& v) const
    {
        return fst->in_edges(v);
    }

    std::vector<scrf_t::edge_type> scrf_t::out_edges(scrf_t::vertex_type const& v) const
    {
        return fst->out_edges(v);
    }

    real scrf_t::weight(scrf_t::edge_type const& e) const
    {
        return (*weight_func)(*fst, e);
    }

    std::string scrf_t::input(scrf_t::edge_type const& e) const
    {
        return fst->input(e);
    }

    std::string scrf_t::output(scrf_t::edge_type const& e) const
    {
        return fst->output(e);
    }

    std::vector<scrf_t::vertex_type> scrf_t::initials() const
    {
        return fst->initials();
    }

    std::vector<scrf_t::vertex_type> scrf_t::finals() const
    {
        return fst->finals();
    }

    namespace feature {

        bias::bias()
        {}

        void bias::operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            feat.class_param[fst.output(e)].push_back(1);
            feat.class_param["shared"].push_back(1);
        }

        length_value::length_value(int max_seg)
            : max_seg(max_seg)
        {}

        void length_value::operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            auto const& lat = *(fst.fst1);
            int tail = lat.tail(std::get<0>(e));
            int head = lat.head(std::get<0>(e));

            int tail_time = lat.data->vertices.at(tail).time;
            int head_time = lat.data->vertices.at(head).time;

            auto& v = feat.class_param[fst.output(e)];

            v.push_back(head_time - tail_time);
            v.push_back(std::pow(head_time - tail_time, 2));
        }

        length_indicator::length_indicator(int max_seg)
            : max_seg(max_seg)
        {}

        void length_indicator::operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            auto const& lat = *(fst.fst1);
            int tail = lat.tail(std::get<0>(e));
            int head = lat.head(std::get<0>(e));

            int tail_time = lat.data->vertices.at(tail).time;
            int head_time = lat.data->vertices.at(head).time;

            auto& v = feat.class_param[fst.output(e)];
            int size = v.size();
            v.resize(size + max_seg + 1);

            if (fst.output(e) != "<s>" && fst.output(e) != "</s>" && fst.output(e) != "sil") {
                v.at(size + std::min(head_time - tail_time, max_seg)) = 1;
            }
        }

        frame_avg::frame_avg(std::vector<std::vector<real>> const& inputs)
            : inputs(inputs)
        {}

        void frame_avg::operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            if (ebt::in(std::get<0>(e), feat_cache)) {
                auto& u = feat_cache.at(std::get<0>(e));
                auto& v = feat.class_param[fst.output(e)];
                v.insert(v.end(), u.begin(), u.end());
                return;
            }

            auto const& lat = *(fst.fst1);
            int tail = lat.tail(std::get<0>(e));
            int head = lat.head(std::get<0>(e));

            int tail_time = std::min<int>(inputs.size() - 1, lat.data->vertices.at(tail).time);
            int head_time = std::min<int>(inputs.size(), lat.data->vertices.at(head).time);

            std::vector<real> avg;
            avg.resize(inputs.front().size());

            if (tail_time < head_time) {
                for (int i = tail_time; i < head_time; ++i) {
                    auto const& v = inputs.at(i);

                    for (int j = 0; j < v.size(); ++j) {
                        avg[j] += v.at(j);
                    }
                }

                for (int j = 0; j < avg.size(); ++j) {
                    avg[j] /= real(head_time - tail_time);
                }
            }

            auto& v = feat.class_param[fst.output(e)];
            v.insert(v.end(), avg.begin(), avg.end());

            feat_cache[std::get<0>(e)] = std::move(avg);
        }

        frame_samples::frame_samples(std::vector<std::vector<real>> const& inputs,
            int samples)
            : inputs(inputs), samples(samples)
        {}

        void frame_samples::operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            auto const& lat = *(fst.fst1);
            int tail = lat.tail(std::get<0>(e));
            int head = lat.head(std::get<0>(e));

            int tail_time = lat.data->vertices.at(tail).time;
            int head_time = lat.data->vertices.at(head).time;

            real span = (head_time - tail_time) / samples;

            auto& v = feat.class_param[fst.output(e)];
            for (int i = 0; i < samples; ++i) {
                auto& u = inputs.at(std::min<int>(std::floor(tail_time + (i + 0.5) * span), inputs.size() - 1));
                v.insert(v.end(), u.begin(), u.end());
            }

        }

        left_boundary::left_boundary(std::vector<std::vector<real>> const& inputs)
            : inputs(inputs)
        {}

        void left_boundary::operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            auto const& lm = *(fst.fst2);

            int lm_tail = lm.tail(std::get<1>(fst.tail(e)));

            std::string lex = lm.data->history.at(lm_tail) + "_" + fst.output(e);

            auto const& lat = *(fst.fst1);
            int tail = lat.tail(std::get<0>(e));
            int head = lat.head(std::get<0>(e));

            int tail_time = lat.data->vertices.at(tail).time;

            if (ebt::in(tail_time, feat_cache)) {
                auto& u = feat_cache.at(tail_time);
                auto& v = feat.class_param[lex];
                v.insert(v.end(), u.begin(), u.end());
                return;
            }

            auto& v = feat.class_param[lex];

            std::vector<real> f;
            for (int i = 0; i < 3; ++i) {
                auto& tail_u = inputs.at(std::min<int>(inputs.size() - 1, std::max<int>(tail_time - i, 0)));
                f.insert(f.end(), tail_u.begin(), tail_u.end());
            }
            v.insert(v.end(), f.begin(), f.end());

            feat_cache[tail_time] = std::move(f);
        }

        right_boundary::right_boundary(std::vector<std::vector<real>> const& inputs)
            : inputs(inputs)
        {}

        void right_boundary::operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            auto const& lm = *(fst.fst2);

            int lm_tail = lm.tail(std::get<1>(fst.tail(e)));

            std::string lex = lm.data->history.at(lm_tail) + "_" + fst.output(e);

            auto const& lat = *(fst.fst1);
            int tail = lat.tail(std::get<0>(e));
            int head = lat.head(std::get<0>(e));

            int head_time = lat.data->vertices.at(head).time;

            if (ebt::in(head_time, feat_cache)) {
                auto& u = feat_cache.at(head_time);
                auto& v = feat.class_param[lex];
                v.insert(v.end(), u.begin(), u.end());
                return;
            }

            auto& v = feat.class_param[lex];

            std::vector<real> f;
            for (int i = 0; i < 3; ++i) {
                auto& tail_u = inputs.at(std::min<int>(head_time + i, inputs.size() - 1));
                f.insert(f.end(), tail_u.begin(), tail_u.end());
            }
            v.insert(v.end(), f.begin(), f.end());

            feat_cache[head_time] = std::move(f);
        }

        void lm_score::operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            feat.class_param["shared"].push_back(fst.fst2->weight(std::get<1>(e)));
        }

        void lattice_score::operator()(
            param_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            feat.class_param["shared"].push_back(fst.fst1->weight(std::get<0>(e)));
        }

    }

    fst::path<scrf_t> shortest_path(scrf_t& s,
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
            }

            result.in_edges[u].push_back(e);

            if (v > int(result.out_edges.size()) - 1) {
                result.out_edges.resize(v + 1);
            }

            result.out_edges[v].push_back(e);

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
    
    std::vector<std::vector<real>> load_features(std::string filename)
    {
        std::vector<std::vector<real>> result;
        std::ifstream ifs { filename };
    
        std::string line;
        while (std::getline(ifs, line)) {
            std::vector<real> vec;
    
            std::vector<std::string> parts = ebt::split(line);
            for (auto& p: parts) {
                vec.push_back(std::stod(p));
            }
    
            result.push_back(vec);
        }
    
        return result;
    }

    std::vector<std::vector<real>> load_features(std::string filename, int nfeat)
    {
        std::vector<std::vector<real>> result;
        std::ifstream ifs { filename };
    
        std::string line;
        while (std::getline(ifs, line)) {
            std::vector<real> vec;
    
            std::vector<std::string> parts = ebt::split(line);
            for (auto& p: parts) {
                if (vec.size() == nfeat) {
                    break;
                }
                vec.push_back(std::stod(p));
            }
    
            result.push_back(vec);
        }
    
        return result;
    }
    
    std::unordered_set<std::string> load_phone_set(std::string filename)
    {
        std::unordered_set<std::string> result;
    
        std::ifstream ifs{filename};
    
        std::string line;
        while (std::getline(ifs, line)) {
            result.insert(line);
        }
    
        return result;
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

    std::shared_ptr<lm::fst> erase_input(std::shared_ptr<lm::fst> lm)
    {
        lm::fst result = *lm;
        result.data = std::make_shared<lm::fst_data>(*(lm->data));
        result.data->in_edges_map.clear();
        result.data->in_edges_map.resize(result.data->edges.size());
        result.data->out_edges_map.clear();
        result.data->out_edges_map.resize(result.data->edges.size());
        for (int e = 0; e < result.data->edges.size(); ++e) {
            auto& e_data = result.data->edges.at(e);
            e_data.input = "<label>";
            int tail = result.tail(e);
            int head = result.head(e);
            result.data->out_edges_map[tail]["<label>"].push_back(e);
            result.data->in_edges_map[head]["<label>"].push_back(e);
        }

        return std::make_shared<lm::fst>(result);
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

    real backoff_cost::operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        return fst.input(e) == "<eps>" ? -1 : 0;
    }

    overlap_cost::overlap_cost(fst::path<scrf_t> const& gold)
        : gold(gold)
    {}

    real overlap_cost::operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        if (ebt::in(std::get<0>(e), edge_cache)) {
            
            int tail = std::get<0>(fst.tail(e));
            int head = std::get<0>(fst.head(e));

            int tail_time = fst.fst1->data->vertices.at(tail).time;
            int head_time = fst.fst1->data->vertices.at(head).time;

            int min_cost = std::numeric_limits<int>::max();

            for (auto& e_g: edge_cache.at(std::get<0>(e))) {
                int gold_tail = std::get<0>(gold.tail(e_g));
                int gold_head = std::get<0>(gold.head(e_g));

                int gold_tail_time = gold.data->base_fst->fst->fst1->data->vertices.at(gold_tail).time;
                int gold_head_time = gold.data->base_fst->fst->fst1->data->vertices.at(gold_head).time;

                int overlap = std::min(gold_head_time, head_time) - std::max(gold_tail_time, tail_time);
                int union_ = std::max(gold_head_time, head_time) - std::min(gold_tail_time, tail_time);
                int cost = (gold.output(e_g) == fst.output(e) ? union_ - overlap : union_);

                if (gold.output(e_g) == "<s>" && fst.output(e) == "<s>"
                        || gold.output(e_g) == "</s>" && fst.output(e) == "</s>"
                        || gold.output(e_g) == "sil" && fst.output(e) == "sil") {

                    cost = head_time - tail_time - overlap;
                }

                if (cost < min_cost) {
                    min_cost = cost;
                }
            }

            if (edge_cache.at(std::get<0>(e)).size() == 0) {
                return head_time - tail_time;
            }

            return min_cost;

        }

        int tail = std::get<0>(fst.tail(e));
        int head = std::get<0>(fst.head(e));

        int tail_time = fst.fst1->data->vertices.at(tail).time;
        int head_time = fst.fst1->data->vertices.at(head).time;

        int max_overlap = 0;
        std::vector<std::tuple<int, int>> max_overlap_edges;

        for (auto& e_g: gold.edges()) {
            int gold_tail = std::get<0>(gold.tail(e_g));
            int gold_head = std::get<0>(gold.head(e_g));

            int gold_tail_time = gold.data->base_fst->fst->fst1->data->vertices.at(gold_tail).time;
            int gold_head_time = gold.data->base_fst->fst->fst1->data->vertices.at(gold_head).time;

            int overlap = std::min(gold_head_time, head_time) - std::max(gold_tail_time, tail_time);

            if (overlap > max_overlap) {
                max_overlap = overlap;
                max_overlap_edges.clear();
                max_overlap_edges.push_back(e_g);
            } else if (overlap == max_overlap) {
                max_overlap_edges.push_back(e_g);
            }
        }

        int min_cost = std::numeric_limits<int>::max();

        for (auto& e_g: max_overlap_edges) {
            int gold_tail = std::get<0>(gold.tail(e_g));
            int gold_head = std::get<0>(gold.head(e_g));

            int gold_tail_time = gold.data->base_fst->fst->fst1->data->vertices.at(gold_tail).time;
            int gold_head_time = gold.data->base_fst->fst->fst1->data->vertices.at(gold_head).time;

            int overlap = std::min(gold_head_time, head_time) - std::max(gold_tail_time, tail_time);
            int union_ = std::max(gold_head_time, head_time) - std::min(gold_tail_time, tail_time);
            int cost = (gold.output(e_g) == fst.output(e) ? union_ - overlap : union_);

            if (gold.output(e_g) == "<s>" && fst.output(e) == "<s>"
                    || gold.output(e_g) == "</s>" && fst.output(e) == "</s>"
                    || gold.output(e_g) == "sil" && fst.output(e) == "sil") {

                cost = head_time - tail_time - overlap;
            }

            if (cost < min_cost) {
                min_cost = cost;
            }
        }

        edge_cache[std::get<0>(e)] = std::move(max_overlap_edges);

        if (max_overlap_edges.size() == 0) {
            return head_time - tail_time;
        }

        return min_cost;
    }

    neg_cost::neg_cost(std::shared_ptr<scrf_weight> cost)
        : cost(cost)
    {}

    real neg_cost::operator()(fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        return -(*cost)(fst, e);
    }

    scrf_t make_graph_scrf(int frames, std::shared_ptr<lm::fst> lm, int max_seg)
    {
        scrf_t result;

        lattice::fst segmentation = make_segmentation_lattice(frames, max_seg);
        lattice::add_eps_loops(segmentation);

        fst::composed_fst<lattice::fst, lm::fst> comp;
        comp.fst1 = std::make_shared<lattice::fst>(segmentation);
        comp.fst2 = lm;

        result.fst = std::make_shared<decltype(comp)>(comp);

        return result;
    }

    hinge_loss::hinge_loss(fst::path<scrf_t> const& gold, scrf_t& graph)
        : gold(gold), graph(graph)
    {
        graph_path = shortest_path(graph, graph.topo_order);
    }

    real hinge_loss::loss()
    {
        real gold_score = 0;

        std::cout << "gold: ";
        for (auto& e: gold.edges()) {
            std::cout << gold.output(e) << " ";
            gold_score += gold.weight(e);
        }
        std::cout << std::endl;

        std::cout << "gold score: " << gold_score << std::endl;

        real graph_score = 0;

        std::cout << "cost aug: ";
        for (auto& e: graph_path.edges()) {
            std::cout << graph.output(e) << " ";
            graph_score += graph_path.weight(e);
        }
        std::cout << std::endl;

        std::cout << "cost aug score: " << graph_score << std::endl; 

        return graph_score - gold_score;
    }

    param_t hinge_loss::param_grad()
    {
        param_t result;

        auto const& gold_feat = *(gold.data->base_fst->feature_func);

        for (auto& e: gold.edges()) {
            param_t p;
            gold_feat(p, *(gold.data->base_fst->fst), e);

            result -= p;
        }

        // std::cout << result.class_param.at("<s>") << std::endl;

        auto const& graph_feat = *(graph.feature_func);

        for (auto& e: graph_path.edges()) {
            param_t p;
            graph_feat(p, *(graph_path.data->base_fst->fst), e);

            result += p;
        }

        // std::cout << "grad of <s>: " << result.class_param.at("<s>") << std::endl;
        // std::cout << "grad of </s>: " << result.class_param.at("</s>") << std::endl;

        return result;
    }

    composite_feature make_feature(
        std::vector<std::string> features,
        std::vector<std::vector<real>> const& inputs, int max_seg)
    {
        composite_feature result;
    
        for (auto& v: features) {
            if (v == "frame-avg") {
                result.features.push_back(std::make_shared<feature::frame_avg>(
                    feature::frame_avg { inputs }));
            } else if (v == "frame-samples") {
                result.features.push_back(std::make_shared<feature::frame_samples>(
                    feature::frame_samples { inputs, 3 }));
            } else if (v == "left-boundary") {
                result.features.push_back(std::make_shared<feature::left_boundary>(
                    feature::left_boundary { inputs }));
            } else if (v == "right-boundary") {
                result.features.push_back(std::make_shared<feature::right_boundary>(
                    feature::right_boundary { inputs }));
            } else if (v == "length-indicator") {
                result.features.push_back(std::make_shared<feature::length_indicator>(
                    feature::length_indicator { max_seg }));
            } else if (v == "length-value") {
                result.features.push_back(std::make_shared<feature::length_value>(
                    feature::length_value { max_seg }));
            } else if (v == "bias") {
                result.features.push_back(std::make_shared<feature::bias>(feature::bias{}));
            } else if (v == "lm-score") {
                result.features.push_back(std::make_shared<feature::lm_score>(feature::lm_score{}));
            } else if (v == "lattice-score") {
                result.features.push_back(std::make_shared<feature::lattice_score>(feature::lattice_score{}));
            } else {
                std::cout << "unknown feature type " << v << std::endl;
                exit(1);
            }
        }
    
        return result;
    }

#if 0
    lattice::fst make_lattice(
        std::vector<std::vector<real>> acoustics,
        std::unordered_set<std::string> phone_set,
        int seg_size)
    {
        lattice::fst_data result;

        result.initial = 0;
        result.final = acoustics.size();

        for (int i = 0; i <= acoustics.size(); ++i) {
            lattice::vertex_data v_data;
            v_data.time = i;
            result.vertices.push_back(v_data);
        }

        for (int i = 0; i <= acoustics.size(); ++i) {
            for (auto& p: phone_set) {
                if (p == "<eps>") {
                    continue;
                }

                for (int j = 1; j <= seg_size && i + j <= acoustics.size(); ++j) {
                    int tail = i;
                    int head = i + j;

                    result.edges.push_back(lattice::edge_data { .label = p,
                        .tail = tail, .head = head });

                    if (std::max(tail, head) >= int(result.in_edges.size()) - 1) {
                        result.in_edges.resize(std::max(tail, head) + 1);
                        result.out_edges.resize(std::max(tail, head) + 1);
                        result.in_edges_map.resize(std::max(tail, head) + 1);
                        result.out_edges_map.resize(std::max(tail, head) + 1);
                    }

                    result.in_edges[head].push_back(int(result.edges.size()) - 1);
                    result.out_edges[tail].push_back(int(result.edges.size()) - 1);

                    result.out_edges_map.at(tail)[p].push_back(result.edges.size() - 1);
                    result.in_edges_map.at(head)[p].push_back(result.edges.size() - 1);
                }
            }
        }

        lattice::fst f;
        f.data = std::make_shared<lattice::fst_data>(std::move(result));
    
        return f;
    }

    void forward_backward_alg::forward_score(scrf const& s)
    {
        alpha.reserve(s.topo_order.size());
        real inf = std::numeric_limits<real>::infinity();

        alpha[s.topo_order.front()] = 0;

        for (int i = 1; i < s.topo_order.size(); ++i) {
            auto const& u = s.topo_order.at(i);

            real value = -std::numeric_limits<real>::infinity();
            for (auto& e: s.in_edges(u)) {
                std::tuple<int, int> v = s.tail(e);
                if (alpha.at(v) != -inf) {
                    value = ebt::log_add(value, alpha.at(v) + s.weight(e));
                }
            }
            alpha[u] = value;
        }
    }

    void forward_backward_alg::backward_score(scrf const& s)
    {
        beta.reserve(s.topo_order.size());
        real inf = std::numeric_limits<real>::infinity();

        beta[s.topo_order.back()] = 0;
        for (int i = s.topo_order.size() - 2; i >= 0; --i) {
            auto const& u = s.topo_order.at(i);

            real value = -std::numeric_limits<real>::infinity();
            for (auto& e: s.out_edges(u)) {
                std::tuple<int, int> v = s.head(e);
                if (beta.at(v) != -inf) {
                    value = ebt::log_add(value, beta.at(v) + s.weight(e));
                }
            }
            beta[u] = value;
        }
    }

    std::unordered_map<std::string, std::vector<real>>
    forward_backward_alg::feature_expectation(scrf const& s)
    {
        /*
        real logZ = alpha.at(s.final());
        auto const& feat_func = *(s.feature_func);
        real inf = std::numeric_limits<real>::infinity();

        std::unordered_map<std::string, std::vector<real>> result;

        for (auto& p2: fst2_edge_index) {
            for (auto& e1: s.fst->fst1->edges()) {
                auto const& feat = feat_func();

                real prob_sum = -inf;

                for (auto& e2: p2.second) {
                    auto e = std::make_tuple(e1, e2);

                    auto tail = s.tail(e);
                    auto head = s.head(e);

                    if (alpha.at(tail) == -inf || beta.at(head) == -inf) {
                        continue;
                    }

                    real prob = alpha.at(tail) + beta.at(head) + s.weight(e) - logZ;
                    prob_sum = ebt::log_add(prob, prob_sum);
                }

                prob_sum = std::exp(prob_sum);

                for (auto& p: feat) {
                    result[p.first].resize(p.second.size());
                    auto& v = result.at(p.first);
                    for (int i = 0; i < p.second.size(); ++i) {
                        v.at(i) += p.second.at(i) * prob_sum;
                    }
                }
            }
        }

        return result;
        */
    }

    log_loss::log_loss(fst::path<scrf> const& gold, scrf const& lat)
        : gold(gold), lat(lat)
    {
        fb.forward_score(lat);
        fb.backward_score(lat);
        std::cout << fb.alpha.at(lat.final()) << " " << fb.beta.at(lat.initial()) << std::endl;
    }

    real log_loss::loss()
    {
        real sum = 0;
        for (auto& e: gold.edges()) {
            sum += gold.weight(e);
        }

        return -sum + fb.alpha.at(lat.final());
    }

    std::unordered_map<std::string, std::vector<real>> const& log_loss::model_grad()
    {
        /*
        result.clear();

        auto const& gold_feat = *(gold.data->base_fst->feature_func);

        for (auto& e: gold.edges()) {
            for (auto& p: gold_feat(e)) {
                result[p.first].resize(p.second.size());
                auto& v = result.at(p.first);
                for (int i = 0; i < p.second.size(); ++i) {
                    v.at(i) -= p.second.at(i);
                }
            }
        }

        auto const& graph_feat = fb.feature_expectation(lat);

        for (auto& p: graph_feat) {
            result[p.first].resize(p.second.size());
            auto& v = result.at(p.first);
            for (int i = 0; i < p.second.size(); ++i) {
                v.at(i) += p.second.at(i);
            }
        }

        return result;
        */
    }

    frame_feature::frame_feature(std::vector<std::vector<real>> const& inputs)
        : inputs(inputs)
    {
        cache.reserve(4000);
    }

    std::unordered_map<std::string, std::vector<real>> const&
    frame_feature::operator()(std::string const& y, int start_time, int end_time) const
    {
        auto k = std::make_tuple(start_time, end_time);

        if (ebt::in(k, cache)) {
            auto& m = cache.at(k);

            std::string ell;

            for (auto& p: m) {
                if (p.first != "shared") {
                    ell = p.first;
                    break;
                }
            }

            std::vector<real> vec = std::move(m.at(ell));

            m[y] = std::move(vec);

            return m;
        }

        std::unordered_map<std::string, std::vector<real>> result;

        real span = (end_time - start_time) / 10;

        auto& vec = result[y];

        for (int i = 0; i < 10; ++i) {
            int frame = std::floor(start_time + (i + 0.5) * span);

            vec.insert(vec.end(), inputs.at(frame).begin(), inputs.at(frame).end());
        }

        cache[k] = std::move(result);

        return cache.at(k);
    }

    frame_score::frame_score(frame_feature const& feat, param_t const& model)
        : feat(feat), model(model)
    {
        cache.reserve(3000000);
    }

    real frame_score::operator()(std::string const& y, int start_time, int end_time) const
    {
        auto k = std::make_tuple(y, start_time, end_time);

        if (ebt::in(k, cache)) {
            return cache.at(k);
        }

        real sum = 0;
        for (auto& p: feat(y, start_time, end_time)) {
            auto const& w = model.weights.at(p.first);

            for (int i = 0; i < p.second.size(); ++i) {
                sum += w.at(i) * p.second.at(i);
            }
        }

        cache[k] = sum;

        return sum;
    }

    linear_score::linear_score(fst::composed_fst<lattice::fst, lm::fst> const& fst,
        frame_score const& f_score)
        : fst(fst), f_score(f_score)
    {
    }

    real linear_score::operator()(std::tuple<int, int> const& e) const
    {
        auto const& lat = *(fst.fst1);

        int tail = lat.tail(std::get<0>(e));
        int head = lat.head(std::get<0>(e));

        int tail_time = lat.data->vertices.at(tail).time;
        int head_time = lat.data->vertices.at(head).time;

        return f_score(fst.output(e), tail_time, head_time);
    }
#endif

}
