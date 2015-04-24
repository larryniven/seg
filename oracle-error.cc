#include "scrf/util.h"
#include "scrf/scrf.h"
#include "scrf/lm.h"
#include "scrf/lattice.h"
#include <fstream>

struct oracle_env {

    std::ifstream lattice_list;
    std::ifstream gold_list;
    std::unordered_set<std::string> phone_set;

    std::unordered_map<std::string, std::string> args;

    oracle_env(std::unordered_map<std::string, std::string> args);

    void run();

};

oracle_env::oracle_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    lattice_list.open(args.at("lattice-list"));
    gold_list.open(args.at("gold-list"));
    phone_set = scrf::load_phone_set(args.at("phone-set"));
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
            data.in_edges_map[0][p1].push_back(data.edges.size() - 1);
            data.out_edges[0].push_back(data.edges.size() - 1);
            data.out_edges_map[0][p1].push_back(data.edges.size() - 1);
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
    lm::fst cost = make_cost(phone_set);

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

        lattice::fst gold = scrf::load_gold(gold_list);

        if (!gold_list) {
            break;
        }

        int gold_edges = gold.edges().size();

        lattice::add_eps_loops(gold, "<eps2>");

        fst::composed_fst<lattice::fst, lm::fst, lattice::fst> comp;
        comp.fst1 = std::make_shared<lattice::fst>(lat);
        comp.fst2 = std::make_shared<lm::fst>(cost);
        comp.fst3 = std::make_shared<lattice::fst>(gold);

        std::vector<std::tuple<int, int, int>> topo_order = comp.vertices();

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

        error_sum += -max;
        length_sum += gold_edges;
        rate_sum += -max / gold_edges;
        density_sum += lat_edges / gold_edges;

        std::cout << ebt::format("error: {} length: {} rate: {} density: {}", -max,
            gold.edges().size(), -max / gold.edges().size(), lat_edges / gold_edges) << std::endl;

        ++i;
    }

    std::cout << ebt::format("total error: {} total length: {} rate: {} avg rate: {} avg density: {}",
        error_sum, length_sum, error_sum / length_sum, rate_sum / i, density_sum / i) << std::endl;
}

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "oracle-error",
        "Calculate oracle error of lattices",
        {
            {"lattice-list", "", true},
            {"gold-list", "", true},
            {"phone-set", "", true}
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
