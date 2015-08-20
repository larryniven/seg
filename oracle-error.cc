#include "scrf/util.h"
#include "scrf/fst.h"
#include "scrf/lm.h"
#include "scrf/lattice.h"
#include "speech/speech.h"
#include <fstream>

struct oracle_env {

    std::ifstream lattice_list;
    std::ifstream gold_list;

    std::vector<std::string> lat_skip;
    std::vector<std::string> gold_skip;

    std::unordered_map<std::string, std::string> args;

    oracle_env(std::unordered_map<std::string, std::string> args);

    void run();

};

oracle_env::oracle_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    lattice_list.open(args.at("lattice-list"));
    gold_list.open(args.at("gold-list"));

    if (ebt::in(std::string("lat-skip"), args)) {
        lat_skip = ebt::split(args.at("lat-skip"), ",");
    }

    if (ebt::in(std::string("gold-skip"), args)) {
        gold_skip = ebt::split(args.at("gold-skip"), ",");
    }
}

lm::fst make_cost(std::unordered_set<std::string> const& phone_set)
{
    lm::fst_data data;

    data.vertices = 1;
    data.initials.push_back(0);
    data.finals.push_back(0);
    data.in_edges.resize(1);
    data.in_edges_map.resize(1);
    data.out_edges.resize(1);
    data.out_edges_map.resize(1);

    for (auto& p1: phone_set) {
        {
            lm::edge_data e_data { 0, 0, -1, p1, "<eps2>"};
            data.edges.push_back(e_data);
            data.in_edges[0].push_back(data.edges.size() - 1);
            data.in_edges_map[0][p1].push_back(data.edges.size() - 1);
            data.out_edges[0].push_back(data.edges.size() - 1);
            data.out_edges_map[0][p1].push_back(data.edges.size() - 1);
        }

        {
            lm::edge_data e_data { 0, 0, -1, "<eps1>", p1};
            data.edges.push_back(e_data);
            data.in_edges[0].push_back(data.edges.size() - 1);
            data.in_edges_map[0]["<eps1>"].push_back(data.edges.size() - 1);
            data.out_edges[0].push_back(data.edges.size() - 1);
            data.out_edges_map[0]["<eps1>"].push_back(data.edges.size() - 1);
        }

        for (auto& p2: phone_set) {
            lm::edge_data e_data { 0, 0, (p1 != p2 ? -1.0 : 0.0), p1, p2 };
            data.edges.push_back(e_data);
            data.in_edges[0].push_back(data.edges.size() - 1);
            data.in_edges_map[0][p1].push_back(data.edges.size() - 1);
            data.out_edges[0].push_back(data.edges.size() - 1);
            data.out_edges_map[0][p1].push_back(data.edges.size() - 1);
        }
    }

    lm::fst result;
    result.data = std::make_shared<lm::fst_data>(std::move(data));

    return result;
}

void oracle_env::run()
{
    int i = 0;

    double error_sum = 0;
    int length_sum = 0;
    double rate_sum = 0;
    double density_sum = 0;

    while (true) {
        lattice::fst lat = lattice::load_lattice(lattice_list);

        if (!lattice_list) {
            break;
        }

        for (auto& e: lat.data->edges) {
            e.weight = 0;
        }

        int lat_edges = lat.edges().size();

        lattice::add_eps_loops(lat, "<eps1>");

        lattice::fst gold = lattice::load_lattice(gold_list);

        if (!gold_list) {
            break;
        }

        for (auto& e: gold.data->edges) {
            e.weight = 0;
        }

        std::unordered_set<std::string> phone_set;

        for (auto& e: lat.edges()) {
            phone_set.insert(lat.output(e));
        }

        for (auto& e: gold.edges()) {
            phone_set.insert(gold.output(e));
        }

        lm::fst cost = make_cost(phone_set);
        for (auto& p: lat_skip) {
            lm::edge_data e_data { 0, 0, 0, p, "<eps2>"};
            auto& data = *(cost.data);
            data.edges.push_back(e_data);
            data.in_edges[0].push_back(data.edges.size() - 1);
            data.in_edges_map[0][p].push_back(data.edges.size() - 1);
            data.out_edges[0].push_back(data.edges.size() - 1);
            data.out_edges_map[0][p].push_back(data.edges.size() - 1);
        }
        for (auto& p: gold_skip) {
            lm::edge_data e_data { 0, 0, 0, "<eps1>", p};
            auto& data = *(cost.data);
            data.edges.push_back(e_data);
            data.in_edges[0].push_back(data.edges.size() - 1);
            data.in_edges_map[0]["<eps1>"].push_back(data.edges.size() - 1);
            data.out_edges[0].push_back(data.edges.size() - 1);
            data.out_edges_map[0]["<eps1>"].push_back(data.edges.size() - 1);
        }

        int gold_edges = 0;
        for (auto& e: gold.edges()) {
            bool skip = false;
            for (auto& p: gold_skip) {
                if (gold.output(e) == p) {
                    skip = true;
                    break;
                }
            }
            if (!skip) {
                ++gold_edges;
            }
        }

        lattice::add_eps_loops(gold, "<eps2>");

        fst::composed_fst<lattice::fst, lm::fst, lattice::fst> comp;
        comp.fst1 = std::make_shared<lattice::fst>(lat);
        comp.fst2 = std::make_shared<lm::fst>(cost);
        comp.fst3 = std::make_shared<lattice::fst>(gold);

        std::vector<std::tuple<int, int, int>> topo_order;
        for (auto& v: lattice::topo_order(lat)) {
            for (auto& u: gold.vertices()) {
                topo_order.push_back(std::make_tuple(v, 0, u));
            }
        }

        fst::one_best<decltype(comp)> one_best;

        for (auto& i: comp.initials()) {
            one_best.extra[i] = { std::make_tuple(-1, -1, -1), 0 };
        }

        one_best.merge(comp, topo_order);

        double max = -std::numeric_limits<double>::infinity();
        for (auto& f: comp.finals()) {
            if (ebt::in(f, one_best.extra)) {
                if (one_best.extra.at(f).value > max) {
                    max = one_best.extra.at(f).value;
                }
            }
        }

        fst::path<decltype(comp)> path = one_best.best_path(comp);

        if (ebt::in(std::string("print-path"), args)) {
            std::cout << lat.data->name << std::endl;

            std::unordered_set<int> vertices;

            for (auto& v: path.vertices()) {
                int lat_v = std::get<0>(v);
                vertices.insert(lat_v);
            }

            std::vector<int> sorted_ver { vertices.begin(), vertices.end() };
            std::sort(sorted_ver.begin(), sorted_ver.end());

            for (auto& v: sorted_ver) {
                std::cout << v << " " << "time=" << lat.data->vertices.at(v).time << std::endl;
            }

            std::cout << "#" << std::endl;

            for (auto& e: path.edges()) {
                int lat_e = std::get<0>(e);

                if (lat.tail(lat_e) == lat.head(lat_e)) {
                    continue;
                }

                std::cout << lat.tail(lat_e) << " " << lat.head(lat_e);

                auto& attr = lat.data->attrs[lat_e];
                for (int i = 0; i < attr.size(); ++i) {
                    auto& p = attr[i];
                    if (i == 0) {
                        std::cout << " ";
                    } else {
                        std::cout << ",";
                    }
                    std::cout << p.first << "=" << p.second;
                }
                std::cout << std::endl;
            }
            std::cout << "." << std::endl;
        } else {
            error_sum += -max;
            length_sum += gold_edges;
            rate_sum += -max / gold_edges;
            density_sum += double(lat_edges) / gold_edges;

            std::cout << ebt::format("error: {} length: {} rate: {} density: {}", -max,
                gold_edges, -max / gold_edges, double(lat_edges) / gold_edges) << std::endl;
        }

        ++i;
    }

    if (!ebt::in(std::string("print-path"), args)) {
        std::cout << ebt::format("total error: {} total length: {} rate: {} avg rate: {} avg density: {}",
            error_sum, length_sum, error_sum / length_sum, rate_sum / i, density_sum / i) << std::endl;
    }
}

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "oracle-error",
        "Calculate oracle error of lattices",
        {
            {"lattice-list", "", true},
            {"gold-list", "", true},
            {"lat-skip", "", false},
            {"gold-skip", "", false},
            {"print-path", "", false}
        }
    };

    if (argc == 1) {
        ebt::usage(spec);
        exit(1);
    }

    auto args = ebt::parse_args(argc, argv, spec);

    std::cout << args << std::endl;

    oracle_env env { args };

    env.run();

    return 0;
}
