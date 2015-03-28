#include "scrf/util.h"
#include "scrf/scrf.h"
#include "scrf/lm.h"
#include "scrf/lattice.h"
#include <fstream>

struct learning_env {

    std::ifstream feature_list;
    std::ifstream gold_list;
    std::shared_ptr<lm::fst> lm;
    int max_seg;
    scrf::scrf_model model;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();
};

learning_env::learning_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    feature_list.open(args.at("feature-list"));
    gold_list.open(args.at("gold-list"));
    lm = std::make_shared<lm::fst>(lm::load_arpa_lm(args.at("lm")));
    max_seg = 10;
    if (ebt::in(std::string("max-seg"), args)) {
        max_seg = std::stoi(args.at("max-seg"));
    }

    model = scrf::load_model(args.at("model"));
}

void learning_env::run()
{
    std::string feature_file;

    std::shared_ptr<lm::fst> lm_output = scrf::erase_input(lm);

    int i = 0;
    while (std::getline(feature_list, feature_file)) {

        std::vector<std::vector<real>> features = scrf::load_features(feature_file);

        scrf::frame_feature frame_feat { features };
        scrf::frame_score frame_s { frame_feat, model };

        lattice::fst gold_lat = scrf::load_gold(gold_list);
        scrf::scrf gold = scrf::make_gold_scrf(gold_lat, lm);
        scrf::backoff_cost bc { *(gold.fst) };
        gold.weight_func = std::make_shared<scrf::backoff_cost>(bc);
        gold.topo_order = gold.vertices();
        fst::path<scrf::scrf> gold_path = scrf::shortest_path(gold, gold.topo_order);

        gold.weight_func = std::make_shared<scrf::linear_score>(scrf::linear_score { *(gold.fst), frame_s });

        scrf::scrf graph = scrf::make_graph_scrf(features.size(), lm_output, max_seg);
        graph.topo_order = graph.vertices();
        graph.weight_func = std::make_shared<scrf::linear_score>(scrf::linear_score { *(graph.fst), frame_s });

        scrf::log_loss loss { gold_path, graph };

        ++i;

        if (i == 10) {
            exit(1);
        }
    }
}

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "learn",
        "Learn segmental CRF",
        {
            {"feature-list", "", true},
            {"gold-list", "", true},
            {"lm", "", true},
            {"max-seg", "", false},
            {"model", "", true}
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
