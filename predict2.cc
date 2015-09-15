#include "scrf/util.h"
#include "scrf/scrf.h"
#include "scrf/lm.h"
#include "scrf/lattice.h"
#include "scrf/scrf_cost.h"
#include "scrf/loss.h"
#include "scrf/scrf_feat.h"
#include "scrf/scrf_weight.h"
#include "scrf/scrf_util.h"
#include "speech/speech.h"
#include <fstream>

struct prediction_env {

    std::ifstream frame_list;
    std::shared_ptr<lm::fst> lm;
    int min_seg;
    int max_seg;
    scrf::param_t param;

    int beam_width;

    std::vector<std::string> features;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

prediction_env::prediction_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    if (ebt::in(std::string("frame-list"), args)) {
        frame_list.open(args.at("frame-list"));
    }

    lm = std::make_shared<lm::fst>(lm::load_arpa_lm(args.at("lm")));

    min_seg = 1;
    if (ebt::in(std::string("min-seg"), args)) {
        min_seg = std::stoi(args.at("min-seg"));
    }

    max_seg = 20;
    if (ebt::in(std::string("max-seg"), args)) {
        max_seg = std::stoi(args.at("max-seg"));
    }

    param = scrf::load_param(args.at("param"));
    features = ebt::split(args.at("features"), ",");    

    if (ebt::in(std::string("beam-width"), args)) {
        beam_width = std::stoi(args.at("beam-width"));
    }
}

void prediction_env::run()
{
    std::string frame_file;

    std::shared_ptr<lm::fst> lm_output = scrf::erase_input(lm);

    int i = 0;
    while (1) {

        std::vector<std::vector<real>> frames;

        if (std::getline(frame_list, frame_file)) {
            frames = speech::load_frames(frame_file);
        }

        if (!frame_list) {
            break;
        }

        scrf::scrf_t graph;

        graph = scrf::make_graph_scrf(frames.size(), lm_output, min_seg, max_seg);

        scrf::composite_feature graph_feat_func = scrf::make_feature2(features, frames);

        graph.weight_func = std::make_shared<scrf::composite_weight>(
            scrf::make_weight(param, graph_feat_func));
        graph.feature_func = std::make_shared<scrf::composite_feature>(graph_feat_func);

        fst::path<scrf::scrf_t> one_best;
        if (ebt::in(std::string("beam-width"), args)) {
            fst::beam_search<scrf::scrf_t> beam_search;
            beam_search.search(graph, beam_width);
            one_best = beam_search.best_path(graph);
        } else {
            one_best = scrf::shortest_path(graph, graph.topo_order);
        }

        double weight = 0;
        for (auto& e: one_best.edges()) {
            std::cout << one_best.output(e) << " ";
            weight += one_best.weight(e);
        }
        std::cout << "(" << frame_file << ")" << std::endl;
        std::cout << "weight: " << weight << std::endl;

        ++i;
    }
}

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "learn",
        "Learn segmental CRF",
        {
            {"frame-list", "", false},
            {"lm", "", true},
            {"max-seg", "", false},
            {"param", "", true},
            {"features", "", true}
        }
    };

    if (argc == 1) {
        ebt::usage(spec);
        exit(1);
    }

    auto args = ebt::parse_args(argc, argv, spec);

    std::cout << args << std::endl;

    prediction_env env { args };

    env.run();

    return 0;
}
