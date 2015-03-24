#include "scrf/scrf.h"
#include <istream>
#include <fstream>
#include "ebt/ebt.h"
#include "opt/opt.h"

namespace scrf {

    scrf_model load_model(std::istream& is)
    {
        scrf_model result;
        std::string line;

        result.weights = ebt::json::json_parser<
            std::unordered_map<std::string, std::vector<double>>>().parse(is);
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
                accu_grad_sq.weights.at(p.first), step_size);
        }
    }

    scrf_weight::~scrf_weight()
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

    double scrf::weight(scrf::edge_type const& e) const
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
