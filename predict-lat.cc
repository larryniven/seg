#include "scrf/util.h"
#include "scrf/scrf.h"
#include "scrf/lm.h"
#include "scrf/lattice.h"
#include "speech/speech.h"
#include "scrf/nn.h"
#include "scrf/loss.h"
#include "scrf/scrf_cost.h"
#include "scrf/scrf_feat.h"
#include "scrf/scrf_weight.h"
#include "scrf/scrf_util.h"
#include "scrf/make_feat.h"
#include <fstream>

struct predict_env {

    std::ifstream frame_batch;
    std::ifstream lattice_batch;
    std::shared_ptr<lm::fst> lm;
    scrf::param_t param;

    std::vector<std::string> features;

    std::unordered_map<std::string, std::string> args;

    predict_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "learn-lat",
        "Learn segmental CRF",
        {
            {"frame-batch", "", false},
            {"lattice-batch", "", true},
            {"lm", "", true},
            {"param", "", true},
            {"features", "", true},
        }
    };

    if (argc == 1) {
        ebt::usage(spec);
        exit(1);
    }

    auto args = ebt::parse_args(argc, argv, spec);

    std::cout << args << std::endl;

    predict_env env { args };

    env.run();

    return 0;
}

predict_env::predict_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    if (ebt::in(std::string("frame-batch"), args)) {
        frame_batch.open(args.at("frame-batch"));
    }

    lattice_batch.open(args.at("lattice-batch"));

    lm = std::make_shared<lm::fst>(lm::load_arpa_lm(args.at("lm")));

    param = scrf::load_param(args.at("param"));

    features = ebt::split(args.at("features"), ",");
}

void predict_env::run()
{
    int i = 1;

    while (1) {

        std::vector<std::vector<real>> frames;

        if (frame_batch) {
            frames = speech::load_frames_batch(frame_batch);
        }

        lattice::fst lat = lattice::load_lattice(lattice_batch);

        if (!lattice_batch) {
            break;
        }

        scrf::composite_feature graph_feat_func = scrf::make_feat(features, frames);

        scrf::scrf_t graph = scrf::make_lat_scrf(lat, lm);

        graph.weight_func =
            graph.weight_func = std::make_shared<scrf::composite_weight>(
                scrf::make_weight(param, graph_feat_func));
        graph.feature_func = std::make_shared<scrf::composite_feature>(graph_feat_func);

        fst::path<scrf::scrf_t> one_best;
        one_best = scrf::shortest_path(graph, graph.topo_order);

        double weight = 0;
        for (auto& e: one_best.edges()) {
            std::cout << one_best.output(e) << " ";
            weight += one_best.weight(e);
        }
        std::cout << "(" << lat.data->name << ")" << std::endl;
        std::cout << "weight: " << weight << std::endl;

#if DEBUG_TOP_10
        if (i == 10) {
            break;
        }
#endif

        ++i;
    }

}

