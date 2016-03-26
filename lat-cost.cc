#include "scrf/lattice.h"
#include "speech/speech.h"
#include "scrf/segcost.h"
#include "scrf/fst.h"
#include <fstream>

struct learning_env {

    std::ifstream lattice_batch;
    std::ifstream ground_truth_batch;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "learn-lat",
        "Learn segmental CRF",
        {
            {"lattice-batch", "", true},
            {"gold-batch", "", true},
            {"print-path", "", false}
        }
    };

    if (argc == 1) {
        ebt::usage(spec);
        exit(1);
    }

    auto args = ebt::parse_args(argc, argv, spec);

    std::cout << args << std::endl;

    learning_env env { args };

    env.run();

    return 0;
}

learning_env::learning_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    lattice_batch.open(args.at("lattice-batch"));
    ground_truth_batch.open(args.at("gold-batch"));
}

void learning_env::run()
{
    double sum_cost = 0;
    int samples = 0;

    while (1) {

        lattice::fst gt_lat = lattice::load_lattice(ground_truth_batch);

        if (!ground_truth_batch) {
            break;
        }

        lattice::fst lat = lattice::load_lattice(lattice_batch);

        if (!lattice_batch) {
            break;
        }

        std::vector<speech::segment> gt_segs;

        for (auto& e: gt_lat.edges()) {
            int tail = gt_lat.tail(e);
            int head = gt_lat.head(e);

            gt_segs.push_back(speech::segment {
                gt_lat.time(tail), gt_lat.time(head), gt_lat.output(e) });
        }

        segcost::overlap_cost cost_func;

        for (auto& e: lat.edges()) {
            int tail = lat.tail(e);
            int head = lat.head(e);

            speech::segment seg { lat.time(tail), lat.time(head), lat.output(e) };

            lat.data->edges[e].weight = -cost_func(gt_segs, seg);
        }

        std::vector<int> topo_order = fst::topo_order(lat);

        fst::forward_one_best<lattice::fst> one_best;
        for (int i: lat.initials()) {
            one_best.extra[i] = {-1, 0};
        }
        one_best.merge(lat, topo_order);
        fst::path<lattice::fst> min_cost_path = one_best.best_path(lat);

        double cost = 0;
        for (auto& e: min_cost_path.edges()) {
            cost += min_cost_path.weight(e);
        }

        if (ebt::in(std::string("print-path"), args)) {
            for (auto& e: min_cost_path.edges()) {
                std::cout << min_cost_path.output(e) << " " << min_cost_path.weight(e) << std::endl;
            }
            std::cout << std::endl;
        }

        std::cout << "cost: " << cost << std::endl;

        sum_cost += cost;
        ++samples;
    }

    std::cout << "avg cost: " << sum_cost / samples << std::endl;
}
