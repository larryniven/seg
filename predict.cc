#include "scrf/util.h"
#include "scrf/scrf.h"
#include "scrf/lm.h"
#include "scrf/lattice.h"
#include "scrf/cost.h"
#include "scrf/loss.h"
#include "scrf/feat.h"
#include "scrf/make_feature.h"
#include "speech/speech.h"
#include "scrf/nn.h"
#include "scrf/weiran.h"
#include <fstream>

struct prediction_env {

    std::ifstream input_list;
    std::ifstream lattice_list;
    std::shared_ptr<lm::fst> lm;
    int max_seg;
    scrf::param_t param;

    std::vector<real> cm_mean;
    std::vector<real> cm_stddev;
    nn::param_t nn_param;
    nn::nn_t nn;

    int beam_width;

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

    if (ebt::in(std::string("lattice-list"), args)) {
        lattice_list.open(args.at("lattice-list"));
    }

    lm = std::make_shared<lm::fst>(lm::load_arpa_lm(args.at("lm")));

    max_seg = 20;
    if (ebt::in(std::string("max-seg"), args)) {
        max_seg = std::stoi(args.at("max-seg"));
    }

    if (ebt::in(std::string("cm-mean"), args)) {
        std::ifstream ifs { args.at("cm-mean") };
        std::string line;
        std::getline(ifs, line);
        std::vector<std::string> parts = ebt::split(line);
        for (auto& v: parts) {
            cm_mean.push_back(std::stod(v));
        }
    }

    if (ebt::in(std::string("cm-stddev"), args)) {
        std::ifstream ifs { args.at("cm-stddev") };
        std::string line;
        std::getline(ifs, line);
        std::vector<std::string> parts = ebt::split(line);
        for (auto& v: parts) {
            cm_stddev.push_back(std::stod(v));
        }
    }

    if (ebt::in(std::string("nn-param"), args)) {
        nn_param = nn::load_param(args.at("nn-param"));
        nn = nn::make_nn(nn_param);
    }

    if (ebt::in(std::string("weiran-nn-param"), args)) {
        nn_param = nn::load_param(args.at("weiran-nn-param"));
        nn = weiran::make_nn(nn_param);
    }

    param = scrf::load_param(args.at("param"));
    features = ebt::split(args.at("features"), ",");    

    if (ebt::in(std::string("beam-width"), args)) {
        beam_width = std::stoi(args.at("beam-width"));
    }
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

        if (ebt::in(std::string("lattice-list"), args)) {
            lattice::fst lat = lattice::load_lattice(lattice_list);

            if (!lattice_list) {
                break;
            }

            lattice::add_eps_loops(lat);

            fst::composed_fst<lattice::fst, lm::fst> comp;
            comp.fst1 = std::make_shared<lattice::fst>(std::move(lat));
            comp.fst2 = lm;
            graph.fst = std::make_shared<decltype(comp)>(comp);

            auto lm_v = lm->vertices();
            std::reverse(lm_v.begin(), lm_v.end());

            std::vector<std::tuple<int, int>> topo_order;
            for (auto& v: lattice::topo_order(*(comp.fst1))) {
                for (auto& u: lm_v) {
                    topo_order.push_back(std::make_tuple(v, u));
                }
            }
            graph.topo_order = std::move(topo_order);
        } else {
            graph = scrf::make_graph_scrf(inputs.size(), lm_output, max_seg);
            graph.topo_order = scrf::topo_order(graph);
        }

        if (!input_list) {
            break;
        }

        scrf::composite_feature graph_feat_func = scrf::make_feature(features, inputs, max_seg,
            cm_mean, cm_stddev, nn);

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
            {"lattice-list", "", false},
            {"lm", "", true},
            {"max-seg", "", false},
            {"param", "", true},
            {"features", "", true},
            {"cm-mean", "", false},
            {"cm-stddev", "", false},
            {"nn-param", "", false},
            {"weiran-nn-param", "", false},
            {"beam-width", "", false}
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
