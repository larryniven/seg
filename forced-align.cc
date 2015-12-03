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
#include "scrf/make_feat.h"
#include <fstream>

struct prediction_env {

    std::ifstream frame_list;
    int min_seg;
    int max_seg;
    scrf::param_t param;

    std::ifstream label_list;

    std::vector<std::string> features;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

prediction_env::prediction_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    frame_list.open(args.at("frame-list"));

    label_list.open(args.at("label-list"));

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
}

void prediction_env::run()
{
    std::string frame_file;
    std::string line;

    int i = 0;
    while (1) {

        std::vector<std::vector<real>> frames;

        if (std::getline(frame_list, frame_file)) {
            frames = speech::load_frames(frame_file);
        }

        if (!frame_list) {
            break;
        }

        std::getline(label_list, line);

        if (!label_list) {
            break;
        }

        std::vector<std::string> labels = ebt::split(line);
        labels.pop_back();

        scrf::scrf_t graph;

        graph = scrf::make_forced_alignment_scrf(frames.size(), labels, min_seg, max_seg);

        scrf::composite_feature graph_feat_func = scrf::make_feat(features, frames);

        graph.weight_func = std::make_shared<scrf::composite_weight>(
            scrf::make_weight(param, graph_feat_func));
        graph.feature_func = std::make_shared<scrf::composite_feature>(graph_feat_func);

        fst::path<scrf::scrf_t> one_best;
        one_best = scrf::shortest_path(graph, graph.topo_order);

        auto& lat = *graph.fst->fst1;

        std::cout << frame_file << std::endl;
        for (auto& v: one_best.vertices()) {
            std::cout << std::get<0>(v)
                << " time=" << lat.data->vertices.at(std::get<0>(v)).time << std::endl;
        }
        std::cout << "#" << std::endl;
        for (auto& e: one_best.edges()) {
            std::cout << std::get<0>(one_best.tail(e)) << " " << std::get<0>(one_best.head(e))
                << " label=" << one_best.output(e) << std::endl;
        }
        std::cout << "." << std::endl;

        ++i;
    }
}

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "forced-align",
        "Align frames to labels with segmental CRF",
        {
            {"frame-list", "", true},
            {"label-list", "", true},
            {"max-seg", "", false},
            {"min-seg", "", false},
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
