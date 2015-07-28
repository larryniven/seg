#include "scrf/util.h"
#include "scrf/scrf.h"
#include "scrf/lm.h"
#include "scrf/lattice.h"
#include "scrf/cost.h"
#include "scrf/loss.h"
#include "scrf/feat.h"
#include "scrf/make_feature.h"
#include "speech/speech.h"
#include <fstream>

struct prediction_env {

    std::ifstream input_list;
    std::shared_ptr<lm::fst> lm;
    int max_seg;
    scrf::param_t param;

    std::vector<std::string> features;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

prediction_env::prediction_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    if (ebt::in(std::string("input-list"), args)) {
        input_list.open(args.at("input-list"));
    }

    lm = std::make_shared<lm::fst>(lm::load_arpa_lm(args.at("lm")));

    max_seg = 20;
    if (ebt::in(std::string("max-seg"), args)) {
        max_seg = std::stoi(args.at("max-seg"));
    }

    param = scrf::load_param(args.at("param"));
    features = ebt::split(args.at("features"), ",");    
}

void prediction_env::run()
{
    std::string input_file;

    std::shared_ptr<lm::fst> lm_output = scrf::erase_input(lm);

    int i = 0;
    while (1) {

        std::vector<std::vector<real>> inputs;

        if (std::getline(input_list, input_file)) {
            inputs = speech::load_frames(input_file);
        }

        scrf::scrf_t graph;

        graph = scrf::make_graph_scrf(inputs.size(), lm_output, max_seg);
        graph.topo_order = scrf::topo_order(graph);

        if (!input_list) {
            break;
        }

        scrf::composite_feature graph_feat_func = scrf::make_feature2(features, inputs);

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
        std::cout << "(" << input_file << ")" << std::endl;
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
            {"input-list", "", false},
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
