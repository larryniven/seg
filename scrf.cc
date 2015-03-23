#include "scrf-first-pass/scrf.h"
#include "nn/neural_net.h"
#include "misc/clarkson_moreno.h"
#include <istream>
#include <fstream>
#include "ebt/ebt.h"
#include "opt/opt.h"

namespace scrf {

    scrf_model load_model(std::istream& is)
    {
        scrf_model result;
        std::string line;

        result.weights = ebt::json::json_parser<std::unordered_map<std::string, double>>().parse(is);
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
        scrf_model& accu_grad_sq, double step_size)
    {
        for (auto& p: model.weights) {
            opt::adagrad_update(p.second, grad.weights.at(p.first),
                accu_grad_sq.at(p.first), step_size);
        }
    }

    std::vector<double> acoustic_feature(std::vector<std::vector<double>> const& acoustics,
        int start, int end)
    {
        std::vector<double> result = clarkson_moreno_feature(acoustics, start, end);

        int center = std::min<int>((start + end) / 2, acoustics.size() - 1);

        std::vector<double> const& center_frame = acoustics.at(center);

        for (int i = 39; i < center_frame.size(); ++i) {
            result.push_back(center_frame.at(i));
        }

        return result;
    }

    scrf_score::~scrf_score()
    {}

    nn_score::nn_score(fst::composed_fst<lattice::fst, lm::fst> const& fst,
        scrf_data const& data, kernel& kern,
        std::unordered_map<std::string, std::vector<std::string>> const& subclasses)
        : fst_(fst), data_(data), kern_(kern), subclasses_(subclasses)
    {}

    double nn_score::operator()(nn_score::edge_type const& e) const
    {
        if (ebt::in(std::get<0>(e), score_cache_)) {
            return score_cache_.at(std::get<0>(e));
        }

        lattice::fst::vertex_type tail = fst_.fst1.tail(std::get<0>(e));
        lattice::fst::vertex_type head = fst_.fst1.head(std::get<0>(e));
        std::string label = fst_.input(e);

        int layer = data_.model.nn_param.weights.size();

        if (ebt::in(std::make_tuple(tail, head), neurons_cache_)) {
            score_cache_[std::get<0>(e)] = neural_net::score(layer, label, data_.model.nn_param,
                neurons_cache_.at(std::make_tuple(tail, head)), subclasses_);
        } else {
            std::vector<double> features = kern_(acoustic_feature(data_.acoustics,
                fst_.fst1.data.vertices.at(tail).time, fst_.fst1.data.vertices.at(head).time));
            neural_net::forward_prop fprop { data_.activation };
            std::vector<std::vector<double>> neurons = fprop(data_.model.nn_param, features);
            neurons_cache_[std::make_tuple(tail, head)] = neurons;

            score_cache_[std::get<0>(e)] = neural_net::score(layer, label, data_.model.nn_param, neurons, subclasses_);
        }

        return score_cache_.at(std::get<0>(e));
    }

    scrf_cost::~scrf_cost()
    {}

    dislike_backoff_cost::dislike_backoff_cost(dislike_backoff_cost::fst_type const& fst)
        : fst_(fst)
    {}

    double dislike_backoff_cost::operator()(dislike_backoff_cost::edge_type const& e) const
    {
        return fst_.input(e) == "<eps>" ? -1 : 0;
    }

    overlap_cost::overlap_cost(overlap_cost::fst_type const& fst,
        fst::path<scrf> const& gold)
        : fst_(fst), gold_(gold)
    {
    }

    double overlap_cost::operator()(overlap_cost::fst_type::edge_type const& e) const
    {
        auto fst_time = [&](overlap_cost::fst_type::vertex_type const& v) {
            return fst_.fst1.data.vertices.at(std::get<0>(v)).time;
        };

        if (fst_time(fst_.tail(e)) == fst_time(fst_.head(e))) {
            return 0;
        }

        if (ebt::in(std::get<0>(e), cost_cache_)) {
            return cost_cache_.at(std::get<0>(e));
        }

        long max = -1;
        lattice::fst::edge_type argmax;

        auto& gold_lat = gold_.data.base_fst.fst.fst1;

        auto gold_time = [&](lattice::fst::vertex_type const& v) {
            return gold_lat.data.vertices.at(v).time;
        };

        if (ebt::in(std::get<0>(e), cache_)) {
            argmax = cache_.at(std::get<0>(e));
        } else {
            for (auto& i: gold_lat.edges()) {
                double overlap = std::max(0L,
                    std::min(gold_time(gold_lat.head(i)), fst_time(fst_.head(e)))
                    - std::max(gold_time(gold_lat.tail(i)), fst_time(fst_.tail(e))));

                if (overlap > max) {
                    max = overlap;
                    argmax = i;
                }
            }

            cache_[std::get<0>(e)] = argmax;
        }

        double overlap =
            std::min(gold_time(gold_lat.head(argmax)), fst_time(fst_.head(e)))
            - std::max(gold_time(gold_lat.tail(argmax)), fst_time(fst_.tail(e)));

        double union_ = 
            std::max(gold_time(gold_lat.head(argmax)), fst_time(fst_.head(e)))
            - std::min(gold_time(gold_lat.tail(argmax)), fst_time(fst_.tail(e)));

        if (gold_lat.input(argmax) == fst_.input(e)) {
            cost_cache_[std::get<0>(e)] = union_ - overlap;
        } else {
            cost_cache_[std::get<0>(e)] = union_;
        }

        return cost_cache_.at(std::get<0>(e));
    }

    std::vector<scrf::vertex_type> scrf::vertices() const
    {
        return fst.vertices();
    }

    std::vector<scrf::edge_type> scrf::edges() const
    {
        return fst.edges();
    }

    scrf::vertex_type scrf::head(scrf::edge_type const& e) const
    {
        return fst.head(e);
    }

    scrf::vertex_type scrf::tail(scrf::edge_type const& e) const
    {
        return fst.tail(e);
    }

    std::vector<scrf::edge_type> scrf::in_edges(scrf::vertex_type const& v) const
    {
        return fst.in_edges(v);
    }

    std::vector<scrf::edge_type> scrf::out_edges(scrf::vertex_type const& v) const
    {
        return fst.out_edges(v);
    }

    double scrf::weight(scrf::edge_type const& e) const
    {
        double sum = 0;
        if (score != nullptr) {
            sum += (*score)(e);
        }
        if (cost != nullptr) {
            sum += (*cost)(e);
        }

        if (!ebt::in(std::get<1>(e), lm_cache_)) {
            lm_cache_[std::get<1>(e)]= ebt::dot(data.model.linear_param, feature(e));
        }
        sum += lm_cache_.at(std::get<1>(e));

        return sum;
    }

    std::string scrf::input(scrf::edge_type const& e) const
    {
        return fst.input(e);
    }

    std::string scrf::output(scrf::edge_type const& e) const
    {
        return fst.output(e);
    }

    scrf::vertex_type scrf::initial() const
    {
        return fst.initial();
    }

    scrf::vertex_type scrf::final() const
    {
        return fst.final();
    }

    ebt::SparseVector scrf::feature(scrf::edge_type const& e) const
    {
        ebt::SparseVector result;

        result("glm") = fst.fst2.weight(std::get<1>(e));
        result("bias") = 1;

        return result;
    }

    void forward_best(fst::lazy_k_best<scrf>& k_best, scrf& scrf_)
    {
        std::vector<scrf::vertex_type> stack;

        for (int i = scrf_.fst.fst1.data.vertices.size() - 1; i >= 0; --i) {
            for (auto& j: scrf_.fst.fst2.vertices()) {
                if (std::make_tuple(i, j) == scrf_.initial()) {
                    continue;
                }

                stack.push_back(std::make_tuple(i, j));
            }
        }

        k_best.extra[scrf_.initial()].deck.push_back({std::make_tuple(-1, -1), -1, 0});

        k_best.merge(scrf_, stack);
    }

    std::unordered_map<scrf::edge_type,
        std::vector<std::vector<double>>>
    path_neurons(fst::path<scrf> const& path, kernel& kern)
    {
        std::unordered_map<scrf::edge_type,
            std::vector<std::vector<double>>> result;

        auto time = [&](scrf::vertex_type const& v) {
            return path.data.base_fst.fst.fst1.data.vertices.at(std::get<0>(v)).time;
        };

        for (auto&& e: path.edges()) {
            auto tail = path.tail(e);
            auto head = path.head(e);
            std::string label = path.input(e);
    
            std::vector<double> features = kern(acoustic_feature(
                path.data.base_fst.data.acoustics, time(tail), time(head)));
            neural_net::forward_prop fprop { path.data.base_fst.data.activation };
            std::vector<std::vector<double>> neurons = fprop(
                path.data.base_fst.data.model.nn_param, features);
            result[e] = neurons;
        }

        return result;
    }

    scrf_model structured_hinge_grad(
        fst::path<scrf> const& gold,
        std::unordered_map<scrf::edge_type,
            std::vector<std::vector<double>>> const& gold_neurons,
        fst::path<scrf> const& c_inf,
        std::unordered_map<scrf::edge_type,
            std::vector<std::vector<double>>> const& c_inf_neurons)
    {
        scrf_model const& model = c_inf.data.base_fst.data.model;

        scrf_model grad;

        grad.nn_param.weights.resize(model.nn_param.weights.size());
        for (int ell = 0; ell < model.nn_param.weights.size(); ++ell) {
            auto& m = model.nn_param.weights.at(ell);
            grad.nn_param.weights.at(ell).resize(m.size());
            for (int i = 0; i < m.size(); ++i) {
                grad.nn_param.weights.at(ell).at(i).resize(m.at(i).size());
            }
        }

        grad.nn_param.pred.resize(model.nn_param.pred.size());

        neural_net::backward_prop bprop { c_inf.data.base_fst.data.activation };
    
        for (auto&& e_i: c_inf.edges()) {
            std::vector<neural_net::loss_func*> losses;

            for (int i = 0; i < model.nn_param.weights.size(); ++i) {
                losses.push_back(new neural_net::empty_loss{i, model.nn_param});
            }
            losses.push_back(new edge_hinge_loss<scrf>{c_inf, e_i, c_inf_neurons.at(e_i)});

            auto w_grad = bprop(model.nn_param, c_inf_neurons.at(e_i), losses);
    
            for (int ell = 0; ell < w_grad.size(); ++ell) {
                auto& m = w_grad.at(ell);
                for (int i = 0; i < m.size(); ++i) {
                    for (int j = 0; j < m.at(i).size(); ++j) {
                        grad.nn_param.weights.at(ell).at(i).at(j) += m.at(i).at(j);
                    }
                }
            }
    
            auto top_grad = losses.back()->model_grad();

            for (auto& p: top_grad) {
                grad.nn_param.pred.back()[p.first].resize(p.second.size());
                for (int i = 0; i < p.second.size(); ++i) {
                    grad.nn_param.pred.back().at(p.first).at(i) += p.second.at(i);
                }
            }

            grad.linear_param += c_inf.data.base_fst.feature(e_i);

            for (auto i: losses) {
                delete i;
            }
        }
    
        for (auto&& e_i: gold.edges()) {
            std::vector<neural_net::loss_func*> losses;

            for (int i = 0; i < model.nn_param.weights.size(); ++i) {
                losses.push_back(new neural_net::empty_loss{i, model.nn_param});
            }
            losses.push_back(new edge_hinge_loss<scrf>{gold, e_i, gold_neurons.at(e_i)});

            auto w_grad = bprop(model.nn_param, gold_neurons.at(e_i), losses);
    
            for (int ell = 0; ell < w_grad.size(); ++ell) {
                auto& m = w_grad.at(ell);
                for (int i = 0; i < m.size(); ++i) {
                    for (int j = 0; j < m.at(i).size(); ++j) {
                        grad.nn_param.weights.at(ell).at(i).at(j) -= m.at(i).at(j);
                    }
                }
            }
    
            auto top_grad = losses.back()->model_grad();

            for (auto& p: top_grad) {
                grad.nn_param.pred.back()[p.first].resize(p.second.size());
                for (int i = 0; i < p.second.size(); ++i) {
                    grad.nn_param.pred.back().at(p.first).at(i) -= p.second.at(i);
                }
            }

            grad.linear_param -= gold.data.base_fst.feature(e_i);

            for (auto i: losses) {
                delete i;
            }
        }

        return grad;
    }
    
    lattice::fst_data load_gold(std::istream& is)
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

        return result;
    }
    
    lattice::fst_data load_gold(std::istream& is, lattice::fst_data const& lat)
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
    
        return result;
    }
    
    std::vector<std::vector<double>> load_acoustics(std::string filename)
    {
        std::vector<std::vector<double>> result;
        std::ifstream ifs{filename};
    
        std::string line;
        while (std::getline(ifs, line)) {
            std::vector<double> vec;
    
            std::vector<std::string> parts = ebt::split(line);
            for (auto& p: parts) {
                vec.push_back(std::stod(p));
            }
    
            result.push_back(vec);
        }
    
        return result;
    }

    std::vector<std::vector<double>> load_acoustics(std::string filename, int nfeat)
    {
        std::vector<std::vector<double>> result;
        std::ifstream ifs{filename};
    
        std::string line;
        while (std::getline(ifs, line)) {
            std::vector<double> vec;
    
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

}
