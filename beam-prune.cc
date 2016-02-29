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
    std::ifstream lattice_batch;
    std::shared_ptr<lm::fst> lm;
    int min_seg;
    int max_seg;
    scrf::param_t param;

    std::vector<std::string> features;

    double alpha;
    std::string output;

    std::unordered_map<std::string, std::string> args;

    pruning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "prune",
        "Prune lattice with segmental CRF",
        {
            {"frame-batch", "", true},
            {"lattice-batch", "", false},
            {"lm", "", true},
            {"max-seg", "", false},
            {"min-seg", "", false},
            {"param", "", true},
            {"features", "", true},
            {"alpha", "", true},
            {"output", "", true}
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
    if (ebt::in(std::string("lattice-batch"), args)) {
        lattice_batch.open(args.at("lattice-batch"));
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

    alpha = std::stod(args.at("alpha"));
    output = args.at("output");
}

void pruning_env::run()
{
    ebt::Timer t;

    std::ofstream ofs { output };

    std::shared_ptr<lm::fst> lm_output = scrf::erase_input(lm);

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

        scrf::composite_feature graph_feat_func = scrf::make_feat(features, frames, {});

        scrf::scrf_t graph;

        if (ebt::in(std::string("lattice-batch"), args)) {
            lattice::fst lat = lattice::load_lattice(lattice_batch);

            if (!lattice_batch) {
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
            for (auto v: lattice::topo_order(*(comp.fst1))) {
                for (auto u: lm_v) {
                    topo_order.push_back(std::make_tuple(v, u));
                }
            }
            graph.topo_order = std::move(topo_order);
        } else {
            graph = scrf::make_graph_scrf(frames.size(), lm_output, min_seg, max_seg);
            graph.topo_order = graph.vertices();
        }
        graph.weight_func = std::make_shared<scrf::composite_weight>(
            scrf::make_weight(features, param, graph_feat_func));
        graph.feature_func = std::make_shared<scrf::composite_feature>(graph_feat_func);

        auto edges = graph.edges();

        lattice::fst& lat = *(graph.fst->fst1);

        lattice::fst_data result;

        std::unordered_map<std::tuple<int, int>, int> vertex_map;
        std::unordered_map<std::tuple<int, int>, int> edge_map;

        std::unordered_set<std::tuple<int, int>> to_expand;
        std::unordered_map<std::tuple<int, int>, double> score;

        auto time = [&](std::tuple<int, int> const& v) {
            return lat.data->vertices.at(std::get<0>(v)).time;
        };

        for (auto& i: graph.initials()) {
            to_expand.insert(i);
            score[i] = 0;

            int id = vertex_map.size();
            vertex_map[i] = id;
            lattice::add_vertex(result, id, time(i));
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
                            lattice::add_vertex(result, id, time(v));
                        }

                        int id = edge_map.size();
                        edge_map[e] = id;
                        lattice::add_edge(result, id, graph.output(e),
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
