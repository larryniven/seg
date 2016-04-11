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

    std::ifstream frame_batch;
    int min_seg;
    int max_seg;
    scrf::first_order::param_t param;

    std::vector<std::string> features;

    std::unordered_map<std::string, int> label_id;
    std::vector<std::string> id_label;
    std::vector<int> labels;
    std::vector<std::vector<int>> label_dim;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "predict-first",
        "Predict segment labels with segmental CRF",
        {
            {"frame-batch", "", false},
            {"max-seg", "", false},
            {"min-seg", "", false},
            {"param", "", true},
            {"features", "", true},
            {"label", "", true},
            {"label-dim", "", false},
            {"length-stat", "", false},
            {"print-seg", "", false}
        }
    };

    if (argc == 1) {
        ebt::usage(spec);
        exit(1);
    }

    auto args = ebt::parse_args(argc, argv, spec);

    for (int i = 0; i < argc; ++i) {
        std::cout << argv[i] << " ";
    }
    std::cout << std::endl;

    prediction_env env { args };

    env.run();

    return 0;
}

prediction_env::prediction_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    if (ebt::in(std::string("frame-batch"), args)) {
        frame_batch.open(args.at("frame-batch"));
    }

    min_seg = 1;
    if (ebt::in(std::string("min-seg"), args)) {
        min_seg = std::stoi(args.at("min-seg"));
    }

    max_seg = 20;
    if (ebt::in(std::string("max-seg"), args)) {
        max_seg = std::stoi(args.at("max-seg"));
    }

    param = scrf::first_order::load_param(args.at("param"));
    features = ebt::split(args.at("features"), ",");    

    label_id = scrf::load_phone_id(args.at("label"));

    id_label.resize(label_id.size());
    for (auto& p: label_id) {
        labels.push_back(p.second);
        id_label[p.second] = p.first;
    }
}

void prediction_env::run()
{
    std::string frame_file;

    int i = 0;

    while (1) {

        std::vector<std::vector<real>> frames;

        frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        scrf::first_order::scrf_t graph;

        graph = scrf::first_order::make_graph_scrf(frames.size(), labels, min_seg, max_seg);

        scrf::first_order::feat_dim_alloc alloc { labels };

        scrf::first_order::composite_feature graph_feat_func
            = scrf::first_order::make_feat(alloc, features, frames, args);

        graph.weight_func = std::make_shared<scrf::first_order::score::linear_score>(
            scrf::first_order::score::linear_score(param,
            std::make_shared<scrf::first_order::composite_feature>(graph_feat_func)));
        graph.feature_func = std::make_shared<scrf::first_order::composite_feature>(graph_feat_func);

        fst::path<scrf::first_order::scrf_t> one_best;

        one_best = scrf::first_order::shortest_path(graph, graph.topo_order);

        if (ebt::in(std::string("print-seg"), args)) {
            auto& lat = *graph.fst;

            std::cout << frame_file << std::endl;

            for (auto& e: one_best.edges()) {
                int tail = lat.tail(e);
                int head = lat.head(e);

                std::cout << "--";

                std::string output = id_label.at(one_best.output(e));
                std::cout << " " << output << std::endl;

                for (int i = 0; i < lat.time(head) - lat.time(tail) - 1; ++i) {
                    std::cout << "|" << std::endl;
                }
            }

            std::cout << "--" << std::endl;

            std::cout << "." << std::endl;
        } else {
            for (auto& e: one_best.edges()) {
                std::cout << id_label[one_best.output(e)] << " ";
            }
            std::cout << "(" << frame_file << ")" << std::endl;

            double weight = 0;

            for (auto& e: one_best.edges()) {
                weight += one_best.weight(e);
            }

            std::cout << "weight: " << weight << std::endl;
        }

        ++i;
    }
}

