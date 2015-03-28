#include "scrf/scrf.h"
#include <istream>
#include <fstream>
#include "ebt/ebt.h"
#include "opt/opt.h"

namespace std {

    size_t hash<tuple<int, int>>::operator()(tuple<int, int> const& t) const
    {
        return std::get<0>(t) * 31 + std::get<1>(t);
    }

}

namespace scrf {

    scrf_model load_model(std::istream& is)
    {
        scrf_model result;
        std::string line;

        result.weights = ebt::json::json_parser<
            std::unordered_map<std::string, std::vector<real>>>().parse(is);
        std::getline(is, line);

        return result;
    }

    scrf_model load_model(std::string filename)
    {
        std::ifstream ifs { filename };
        return load_model(ifs);
    }

    void save_model(std::ostream& os, scrf_model const& model)
    {
        os << model.weights << std::endl;
    }

    void save_model(std::string filename, scrf_model const& model)
    {
        std::ofstream ofs { filename };
        save_model(ofs, model);
    }

    void adagrad_update(scrf_model& model, scrf_model const& grad,
        scrf_model& accu_grad_sq, real step_size)
    {
        for (auto& p: model.weights) {
            opt::adagrad_update(p.second, grad.weights.at(p.first),
                accu_grad_sq.weights.at(p.first), step_size);
        }
    }

    scrf_weight::~scrf_weight()
    {}

    scrf_feature::~scrf_feature()
    {}

    std::vector<scrf::vertex_type> scrf::vertices() const
    {
        return fst->vertices();
    }

    std::vector<scrf::edge_type> scrf::edges() const
    {
        return fst->edges();
    }

    scrf::vertex_type scrf::head(scrf::edge_type const& e) const
    {
        return fst->head(e);
    }

    scrf::vertex_type scrf::tail(scrf::edge_type const& e) const
    {
        return fst->tail(e);
    }

    std::vector<scrf::edge_type> scrf::in_edges(scrf::vertex_type const& v) const
    {
        return fst->in_edges(v);
    }

    std::vector<scrf::edge_type> scrf::out_edges(scrf::vertex_type const& v) const
    {
        return fst->out_edges(v);
    }

    real scrf::weight(scrf::edge_type const& e) const
    {
        return (*weight_func)(e);
    }

    std::string scrf::input(scrf::edge_type const& e) const
    {
        return fst->input(e);
    }

    std::string scrf::output(scrf::edge_type const& e) const
    {
        return fst->output(e);
    }

    scrf::vertex_type scrf::initial() const
    {
        return fst->initial();
    }

    scrf::vertex_type scrf::final() const
    {
        return fst->final();
    }

    fst::path<scrf> shortest_path(scrf& s,
        std::vector<std::tuple<int, int>> const& order)
    {
        fst::one_best<scrf> best;

        best.extra[s.initial()] = {std::make_tuple(-1, -1), 0};

        best.merge(s, order);

        return best.best_path(s);
    }

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

    lattice::fst load_gold(std::istream& is)
    {
        std::string line;
        std::getline(is, line);
    
        lattice::fst_data result;

        int v = 0;
        result.initial = v;
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

        result.final = v;

        lattice::fst f;
        f.data = std::make_shared<lattice::fst_data>(std::move(result));

        return f;
    }
    
    lattice::fst load_gold(std::istream& is, lattice::fst_data const& lat)
    {
        std::string line;
        std::getline(is, line);
    
        lattice::fst_data result;
    
        std::vector<std::tuple<long, long, std::string>> edges;
    
        while (std::getline(is, line) && line != ".") {
            std::vector<std::string> parts = ebt::split(line);
    
            long tail_time = std::min<long>(long(std::stol(parts.at(0)) / 1e5),
                lat.vertices.at(lat.final).time);
            long head_time = std::min<long>(long(std::stol(parts.at(1)) / 1e5),
                lat.vertices.at(lat.final).time);
    
            if (tail_time == head_time) {
                continue;
            }
    
            edges.push_back(std::make_tuple(tail_time, head_time, parts.at(2)));
        }
    
        int i = edges.size() - 1;
        int v = lat.final;
    
        result.final = v;
        result.vertices.resize(v + 1);
        result.vertices.at(v).time = std::get<1>(edges.at(i));
    
        while (i >= 0) {
            bool found = false;
    
            for (auto& e: lat.in_edges.at(v)) {
                int tail = lat.edges.at(e).tail;
    
                if (std::get<0>(edges.at(i)) == lat.vertices.at(tail).time
                        && std::get<2>(edges.at(i)) == lat.edges.at(e).label) {
    
                    if (e >= result.edges.size()) {
                        result.edges.resize(e + 1);
                    }

                    if (v > int(result.in_edges.size()) - 1) {
                        result.in_edges.resize(v + 1);
                    }
    
                    result.in_edges[v].push_back(e);

                    if (tail > int(result.out_edges.size()) - 1) {
                        result.out_edges.resize(tail + 1);
                    }

                    result.out_edges[tail].push_back(e);
    
                    result.edges.at(e).head = v;
                    result.edges.at(e).tail = tail;
                    result.edges.at(e).label = std::get<2>(edges.at(i));
    
                    if (tail >= result.vertices.size()) {
                        result.vertices.resize(tail + 1);
                    }
    
                    result.vertices.at(tail).time = std::get<0>(edges.at(i));
    
                    v = tail;
                    --i;
                    found = true;
                    break;
                }
            }
    
            if (!found) {
                for (auto& e: lat.in_edges.at(v)) {
                    int tail = lat.edges.at(e).tail;
    
                    std::cout << lat.vertices.at(tail).time
                        << " " << lat.vertices.at(v).time
                        << " " << lat.edges.at(e).label << std::endl;
                }
    
                std::cerr << "unable to find \"" << std::get<0>(edges.at(i))
                    << " " << std::get<1>(edges.at(i))
                    << " " << std::get<2>(edges.at(i))
                    << "\" in scrf graph"<< std::endl;
                exit(1);
            }
        }
    
        result.initial = v;

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

    backoff_cost::backoff_cost(fst::composed_fst<lattice::fst, lm::fst> const& fst)
        : fst(fst)
    {}

    real backoff_cost::operator()(std::tuple<int, int> const& e) const
    {
        return fst.input(e) == "<eps>" ? -1 : 0;
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

    scrf make_gold_scrf(lattice::fst gold_lat,
        std::shared_ptr<lm::fst> lm)
    {
        gold_lat.data = std::make_shared<lattice::fst_data>(*(gold_lat.data));
        lattice::add_eps_loops(gold_lat);
        fst::composed_fst<lattice::fst, lm::fst> gold_lm_lat;
        gold_lm_lat.fst1 = std::make_shared<lattice::fst>(std::move(gold_lat));
        gold_lm_lat.fst2 = lm;

        scrf gold;
        gold.fst = std::make_shared<decltype(gold_lm_lat)>(gold_lm_lat);

        return gold;
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

        for (int i = 0; i < frames + 1 - max_seg; ++i) {
            for (int j = 1; j <= max_seg; ++j) {
                int tail = i;
                int head = i + j;

                data.edges.push_back(lattice::edge_data {"<label>", tail, head});
                int e = data.edges.size() - 1;

                data.in_edges.at(head).push_back(e);
                data.in_edges_map.at(head)["<label>"].push_back(e);
                data.out_edges.at(tail).push_back(e);
                data.in_edges_map.at(tail)["<label>"].push_back(e);
            }
        }

        data.initial = 0;
        data.final = frames;

        lattice::fst f;
        f.data = std::make_shared<lattice::fst_data>(std::move(data));

        return f;
    }

    scrf make_graph_scrf(int frames, std::shared_ptr<lm::fst> lm, int max_seg)
    {
        scrf result;

        lattice::fst segmentation = make_segmentation_lattice(frames, max_seg);
        lattice::add_eps_loops(segmentation);

        fst::composed_fst<lattice::fst, lm::fst> comp;
        comp.fst1 = std::make_shared<lattice::fst>(segmentation);
        comp.fst2 = lm;

        result.fst = std::make_shared<decltype(comp)>(comp);

        return result;
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

    frame_score::frame_score(frame_feature const& feat, scrf_model const& model)
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

    namespace detail {

        void save_model(std::ostream& os, model_vector const& model)
        {
            os << model.class_weights << std::endl;
        }

        void save_model(std::string filename, model_vector const& model)
        {
            std::ofstream ofs { filename };

            save_model(ofs, model);
        }

        model_vector load_model(std::istream& is)
        {
            model_vector result;

            ebt::json::json_parser<decltype(result.class_weights)> weights_parser;

            result.class_weights = weights_parser.parse(is);

            std::string line;
            std::getline(is, line);

            return result;
        }

        model_vector load_model(std::string filename)
        {
            std::ifstream ifs { filename };

            return load_model(ifs);
        }

        /*

             scrf
             ====

         */
        scrf::scrf(std::vector<std::vector<real>> const& inputs,
            model_vector const& weights, int labels)
            : inputs(inputs), weights(weights), labels(labels)
        {}

        real scrf::score(int y, int start_time, int end_time) const
        {
            auto const& feat = feature(y, start_time, end_time);

            real sum = 0;
            for (int i = 0; i < feat.size(); ++i) {
                if (feat.size() == 0) {
                    continue;
                }

                auto& u = weights.class_weights.at(i);
                auto& v = feat.at(i);

                for (int j = 0; j < v.size(); ++j) {
                    sum += u.at(j) * v.at(j); 
                }
            }

            return sum;
        }

        real scrf::score(int y1, int y2, int start_time, int end_time) const
        {
            return score(y2, start_time, end_time);
        }

        std::vector<std::vector<real>> const&
        scrf::feature(int y, int start_time, int end_time) const
        {
            feat.clear();
            feat.resize(labels + 1);

            real span = (end_time - start_time) / 3;
            auto& u = feat.at(y);

            for (int i = 0; i < 3; ++i) {
                int center = std::floor(start_time + span * (i + 0.5));

                auto const& v = inputs.at(center);

                u.insert(u.end(), v.begin(), v.end());
            }

            feat[labels].push_back(1);

            return feat;
        }

        /*

            gold_scrf
            =========

         */
        gold_scrf::gold_scrf(std::vector<std::vector<real>> const& inputs,
            model_vector const& model, int labels)
            : scrf(inputs, model, labels)
        {
        }

        /*

            graph_scrf
            ==========

         */
        graph_scrf::graph_scrf(std::vector<std::vector<real>> const& inputs,
            model_vector const& weights, int labels,
            std::unordered_map<std::string, int> const& phone_map,
            int frames, int max_seg)
            : scrf(inputs, weights, labels), phone_map(phone_map)
            , frames(frames), max_seg(max_seg)
        {
            real inf = std::numeric_limits<real>::infinity();

            score_cache.resize(labels);
            for (auto& m: score_cache) {
                m.resize(frames + 1);
                for (auto& v: m) {
                    v.resize(frames + 1, -inf);
                }
            }

            feature_cache.resize(labels);
            for (auto& m: feature_cache) {
                m.resize(frames + 1);
                for (auto& v: m) {
                    v.resize(frames + 1);
                }
            }
        }

        std::vector<std::vector<real>> const&
        graph_scrf::feature(int y, int start_time, int end_time) const
        {
            auto& cache = feature_cache[y][start_time][end_time];

            if (cache.size() != 0) {
                return cache;
            }

            auto const& f = scrf::feature(y, start_time, end_time);

            cache = f;

            return f;
        }

        real graph_scrf::score(int y, int start_time, int end_time) const
        {
            real inf = std::numeric_limits<real>::infinity();

            real& cache = score_cache[y][start_time][end_time];
            if (cache != -inf) {
                 return cache;
            }

            real sum = scrf::score(y, start_time, end_time);

            cache = sum;

            return sum;
        }

        real graph_scrf::score(int y1, int y2, int start_time, int end_time) const
        {
            return scrf::score(y1, y2, start_time, end_time);
        }

        /*

            forward_backward_alg
            ====================

         */
        void forward_backward_alg::forward_score()
        {
            real inf = std::numeric_limits<real>::infinity();

            alpha.resize(model.frames + 1);
            for (int end_time = 0; end_time <= model.frames; ++end_time) {
                alpha[end_time].resize(model.labels, -inf);
            }

            alpha[0][model.phone_map.at("<s>")] = 0;

            for (int end_time = 1; end_time <= model.frames; ++end_time) {

                for (int y2 = 0; y2 < model.labels; ++y2) {

                    real value = -std::numeric_limits<real>::infinity();

                    for (int y1 = 0; y1 < model.labels; ++y1) {
                        for (int seg_len = 1; seg_len <= model.max_seg; ++seg_len) {

                            int start_time = end_time - seg_len;

                            if (start_time < 0 || alpha[start_time][y1] == -inf) {
                                continue;
                            }

                            value = ebt::log_add(value, alpha[start_time][y1]
                                + model.score(y1, y2, start_time, end_time));
                        }
                    }

                    alpha[end_time][y2] = value;
                }
            }
        }

        void forward_backward_alg::backward_score()
        {
            real inf = std::numeric_limits<real>::infinity();

            beta.resize(model.frames + 1);
            for (int start_time = model.frames; start_time >= 0; --start_time) {
                beta[start_time].resize(model.labels, -inf);
            }

            beta[model.frames][model.phone_map.at("</s>")] = 0;

            for (int start_time = model.frames - 1; start_time >= 0; --start_time) {

                for (int y1 = 0; y1 < model.labels; ++y1) {

                    real value = -std::numeric_limits<real>::infinity();

                    for (int y2 = 0; y2 < model.labels; ++y2) {
                        for (int seg_len = 1; seg_len <= model.max_seg; ++seg_len) {

                            int end_time = start_time + seg_len;

                            if (end_time > model.frames || beta[end_time][y2] == -inf) {
                                continue;
                            }

                            value = ebt::log_add(value, beta[end_time][y2]
                                + model.score(y1, y2, start_time, end_time));
                        }
                    }

                    beta[start_time][y1] = value;
                }
            }
        }

        std::vector<std::vector<real>>
        forward_backward_alg::feature_expectation()
        {
            real inf = std::numeric_limits<real>::infinity();
            real logZ = beta[0][model.phone_map.at("<s>")];

            std::vector<std::vector<real>> result;

            for (int start_time = 0; start_time < model.frames; ++start_time) {

                for (int seg_len = 1; seg_len <= model.max_seg; ++seg_len) {
                    int end_time = start_time + seg_len;

                    if (end_time > model.frames) {
                        continue;
                    }

                    for (int y2 = 0; y2 < model.labels; ++y2) {
                        auto const& feat = model.feature(y2, start_time, end_time);

                        result.resize(feat.size());

                        real prob_sum = -inf;

                        for (int y1 = 0; y1 < model.labels; ++y1) {
                            real s = model.score(y1, y2, start_time, end_time);

                            if (alpha[start_time][y1] == -inf || beta[end_time][y2] == -inf) {
                                continue;
                            }

                            real prob = alpha[start_time][y1] + beta[end_time][y2] + s - logZ;

                            prob_sum = ebt::log_add(prob_sum, prob);

                        }

                        prob_sum = std::exp(prob_sum);

                        for (int i = 0; i < feat.size(); ++i) {
                            auto& u = feat.at(i);
                            auto& v = result.at(i);

                            v.resize(std::max(v.size(), u.size()));
                            for (int j = 0; j < u.size(); ++j) {
                                v.at(j) += u.at(j) * prob_sum;
                            }
                        }
                    }
                }
            }

            return result;
        }

        /*

            log_loss
            ========

         */
        log_loss::log_loss(gold_scrf const& gold, graph_scrf const& graph)
            : gold(gold), graph(graph), fb { graph }
        {
            fb.forward_score();
            fb.backward_score();

            std::cout << fb.beta[0][graph.phone_map.at("<s>")] << " "
                << fb.alpha[graph.frames][graph.phone_map.at("</s>")] << std::endl;

        }

        real log_loss::loss()
        {
            real sum = 0;
            for (auto& e: gold.edges) {
                sum -= gold.score(e.label, e.start_time, e.end_time);
            }
            sum += fb.beta[0][graph.phone_map.at("<s>")];

            return sum;
        }

        std::vector<std::vector<real>>
        log_loss::model_grad()
        {
            std::vector<std::vector<real>> result = fb.feature_expectation();

            auto prev = gold.edges.front();
            for (int i = 0; i < gold.edges.size(); ++i) {
                auto& e = gold.edges.at(i);

                auto const& f = gold.feature(e.label, e.start_time, e.end_time);

                for (int j = 0; j < f.size(); ++j) {
                    auto& u = result.at(j);
                    auto const& v = f.at(j);

                    u.resize(std::max(u.size(), v.size()));

                    for (int k = 0; k < v.size(); ++k) {
                        u.at(k) -= v.at(k);
                    }

                }

                prev = e;
            }

            return result;
        }

        std::vector<gold_scrf::edge> load_gold(std::istream& is,
            std::unordered_map<std::string, int> const& phone_map,
            int frames)
        {
            std::string line;

            std::getline(is, line);

            std::vector<gold_scrf::edge> result;

            while (std::getline(is, line) && line != ".") {
                auto parts = ebt::split(line);

                result.push_back(gold_scrf::edge { std::min<int>(frames - 1, int(std::stoi(parts[0]) / 1e5)),
                    std::min<int>(frames, int(std::stoi(parts[1])) / 1e5), phone_map.at(parts[2]) });
            }

            return result;
        }

        std::unordered_map<std::string, int>
        load_phone_map(std::string filename)
        {
            std::ifstream ifs { filename };
            std::string line;

            std::unordered_map<std::string, int> result;

            int i = 0;
            while (std::getline(ifs, line)) {
                result[line] = i++;
            }

            return result;
        }

        std::vector<std::string>
        make_inv_phone_map(std::unordered_map<std::string, int> const& phone_map)
        {
            std::vector<std::string> result;

            result.resize(phone_map.size());

            for (auto& p: phone_map) {
                result[p.second] = p.first;
            }

            return result;
        }

    }

}
