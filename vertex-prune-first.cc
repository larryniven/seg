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

    real alpha;
    std::string output;

    std::unordered_map<std::string, int> label_id;
    std::vector<std::string> id_label;
    std::vector<int> labels;
    std::vector<std::vector<int>> label_dim;

    std::unordered_map<std::string, std::string> args;

    pruning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "vertex-prune-first",
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
            {"label-dim", "", false}
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

    if (ebt::in(std::string("label-dim"), args)) {
        label_dim = scrf::first_order::load_label_dim(args.at("label-dim"), label_id);
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
            = scrf::first_order::make_feat(alloc, features, frames, label_dim);

        scrf::first_order::scrf_t graph = scrf::first_order::make_graph_scrf(frames.size(),
            labels, min_seg, max_seg);

        graph.weight_func = std::make_shared<scrf::first_order::score::cached_linear_score>(
            scrf::first_order::score::cached_linear_score(param,
                std::make_shared<scrf::first_order::composite_feature>(graph_feat_func)));
        graph.feature_func = std::make_shared<scrf::first_order::composite_feature>(graph_feat_func);

        auto edges = graph.edges();

        auto order = graph.topo_order;

        fst::forward_one_best<scrf::first_order::scrf_t> forward;
        for (auto v: graph.initials()) {
            forward.extra[v] = {-1, 0};
        }
        forward.merge(graph, order);

        std::reverse(order.begin(), order.end());
        fst::backward_one_best<scrf::first_order::scrf_t> backward;
        for (auto v: graph.finals()) {
            backward.extra[v] = {-1, 0};
        }
        backward.merge(graph, order);

        real inf = std::numeric_limits<real>::infinity();

        auto fb_alpha = [&](int v) {
            if (ebt::in(v, forward.extra)) {
                return forward.extra[v].value;
            } else {
                return -inf;
            }
        };

        auto fb_beta = [&](int v) {
            if (ebt::in(v, forward.extra)) {
                return backward.extra[v].value;
            } else {
                return -inf;
            }
        };

        real b_max = -inf;

        for (auto& i: graph.initials()) {
            if (fb_beta(i) > b_max) {
                b_max = fb_beta(i);
            }
        }

        real f_max = -inf;

        for (auto& f: graph.finals()) {
            if (fb_alpha(f) > f_max) {
                f_max = fb_alpha(f);
            }
        }

        std::cout << "frames: " << frames.size() << std::endl;
        std::cout << "forward: " << f_max << " backward: " << b_max << std::endl;

        lattice::fst_data result;

        std::unordered_map<int, int> vertex_map;
        std::unordered_map<int, int> edge_map;

        std::vector<int> vertices = graph.vertices();

        double min = inf;
        double max = -inf;

        for (auto& v: vertices) {
            double s = fb_alpha(v) + fb_beta(v);

            if (s < min) {
                min = s;
            }

            if (s > max) {
                max = s;
            }
        }

        for (auto& v: vertices) {
            if (fb_alpha(v) + fb_beta(v) > min + (max - min) * alpha) {
                int id = vertex_map.size();
                vertex_map[v] = id;
                lattice::add_vertex(result, id, graph.fst->time(v));
            }
        }

        for (auto& p: vertex_map) {
            for (auto e: graph.out_edges(p.first)) {
                if (ebt::in(graph.head(e), vertex_map)) {
                    int id = edge_map.size();
                    edge_map[e] = id;
                    lattice::add_edge(result, id, id_label[graph.output(e)],
                        vertex_map.at(graph.tail(e)), vertex_map.at(graph.head(e)), graph.weight(e));
                }
            }
        }

        lattice::fst result_fst;
        result_fst.data = std::make_shared<lattice::fst_data>(result);

        ofs << i << ".lat" << std::endl;

        for (int i = 0; i < result_fst.vertices().size(); ++i) {
            ofs << i << " "
                << "time=" << result_fst.time(i) << std::endl;
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
