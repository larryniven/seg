#include "scrf/util.h"
#include "scrf/scrf_feat.h"
#include "scrf/scrf.h"
#include "scrf/lm.h"
#include "scrf/lattice.h"
#include "speech/speech.h"
#include "scrf/scrf_weight.h"
#include "scrf/scrf_util.h"
#include "scrf/make_feat.h"
#include <fstream>

struct pruning_env {

    std::ifstream frame_batch;
    int min_seg;
    int max_seg;
    scrf::first_order::param_t param;

    std::vector<std::string> features;

    double alpha;
    std::string output;

    std::unordered_map<std::string, int> label_id;
    std::vector<std::string> id_label;
    std::vector<int> labels;
    std::vector<int> id_dim;

    std::unordered_map<std::string, std::string> args;

    pruning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "beam-prune-first",
        "Prune lattice with segmental CRF",
        {
            {"frame-batch", "", true},
            {"max-seg", "", false},
            {"min-seg", "", false},
            {"param", "", true},
            {"features", "", true},
            {"alpha", "", true},
            {"output", "", true},
            {"label", "", true},
            {"logprob-label", "", false}
        }
    };

    if (argc == 1) {
        ebt::usage(spec);
        exit(1);
    }

    auto args = ebt::parse_args(argc, argv, spec);

    std::cout << args << std::endl;

    pruning_env env { args };

    env.run();

    return 0;
}

pruning_env::pruning_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    frame_batch.open(args.at("frame-batch"));

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

    alpha = std::stod(args.at("alpha"));
    output = args.at("output");

    label_id = scrf::load_phone_id(args.at("label"));

    id_label.resize(label_id.size());
    for (auto& p: label_id) {
        labels.push_back(p.second);
        id_label[p.second] = p.first;
    }

    if (ebt::in(std::string("logprob-label"), args)) {
        std::unordered_map<std::string, int> logprob_label
            = scrf::load_phone_id(args.at("logprob-label"));

        id_dim.resize(id_label.size());
        for (int i = 0; i < id_label.size(); ++i) {
            if (i == label_id.at("<s>")) {
                id_dim[i] = logprob_label.at("sil");
            } else if (i == label_id.at("</s>")) {
                id_dim[i] = logprob_label.at("sil");
            } else {
                id_dim[i] = logprob_label.at(id_label[i]);
            }
        }
    }
}

void pruning_env::run()
{
    ebt::Timer t;

    std::ofstream ofs { output };

    int i = 0;

    while (1) {

        std::vector<std::vector<real>> frames;

        if (frame_batch) {
            frames = speech::load_frame_batch(frame_batch);

            if (!frame_batch) {
                break;
            }
        }

        std::cout << i << ".lat" << std::endl;

        scrf::first_order::feat_dim_alloc alloc { labels };

        scrf::first_order::composite_feature graph_feat_func
            = scrf::first_order::make_feat(alloc, features, frames, id_dim);

        scrf::first_order::scrf_t graph = scrf::first_order::make_graph_scrf(frames.size(),
            labels, min_seg, max_seg);

        graph.weight_func = std::make_shared<scrf::first_order::score::cached_linear_score>(
            scrf::first_order::score::cached_linear_score(param,
                std::make_shared<scrf::first_order::composite_feature>(graph_feat_func)));
        graph.feature_func = std::make_shared<scrf::first_order::composite_feature>(graph_feat_func);

        lattice::fst_data result;

        std::unordered_map<int, int> vertex_map;
        std::unordered_map<int, int> edge_map;

        std::unordered_set<int> to_expand;
        std::unordered_map<int, double> score;

        for (auto& i: graph.initials()) {
            to_expand.insert(i);
            score[i] = 0;

            int id = vertex_map.size();
            vertex_map[i] = id;
            lattice::add_vertex(result, id, graph.fst->time(i));
        }

        double inf = std::numeric_limits<double>::infinity();

        for (int t = 0; t < graph.topo_order.size(); ++t) {

            auto u = graph.topo_order[t];

            if (ebt::in(u, to_expand)) {

                to_expand.erase(u);

                double max = -inf;
                double min = inf;

                for (auto e: graph.out_edges(u)) {
                    auto v = graph.head(e);

                    double candidate = ebt::get(score, u, -inf) + graph.weight(e);

                    if (candidate > max) {
                        max = candidate;
                    }

                    if (candidate < min) {
                        min = candidate;
                    }
                }

                for (auto e: graph.out_edges(u)) {
                    auto v = graph.head(e);

                    double candidate = ebt::get(score, u, -inf) + graph.weight(e);

                    if (candidate > min + (max - min) * alpha) {
                        if (!ebt::in(v, vertex_map)) {
                            int id = vertex_map.size();
                            vertex_map[v] = id;
                            lattice::add_vertex(result, id, graph.fst->time(v));
                        }

                        int id = edge_map.size();
                        edge_map[e] = id;
                        lattice::add_edge(result, id, id_label[graph.output(e)],
                            vertex_map.at(graph.tail(e)), vertex_map.at(graph.head(e)), graph.weight(e));

                        if (candidate > ebt::get(score, v, -inf)) {
                            score[v] = candidate;
                            to_expand.insert(v);
                        }
                    }
                }
            }
        }

        lattice::fst result_fst;
        result_fst.data = std::make_shared<lattice::fst_data>(result);

        ofs << i << ".lat" << std::endl;

        for (int i = 0; i < result_fst.vertices().size(); ++i) {
            ofs << i << " "
                << "time=" << result_fst.data->vertices.at(i).time << std::endl;
        }

        ofs << "#" << std::endl;

        for (int e = 0; e < result.edges.size(); ++e) {
            int tail = result_fst.tail(e);
            int head = result_fst.head(e);

            ofs << tail << " " << head << " "
                << "label=" << result_fst.output(e) << ";"
                << "weight=" << result.edges.at(e).weight << std::endl;
        }
        ofs << "." << std::endl;

        auto edges = graph.edges();

        std::cout << "edges: " << edges.size() << " left: " << result.edges.size()
            << " (" << real(result.edges.size()) / edges.size() << ")" << std::endl;

        std::cout << std::endl;

        ++i;

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            exit(1);
        }
#endif

    }
}
