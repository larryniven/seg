#include "scrf/util.h"
#include "scrf/scrf_feat.h"
#include "scrf/scrf.h"
#include "scrf/lm.h"
#include "scrf/lattice.h"
#include "speech/speech.h"
#include "scrf/scrf_weight.h"
#include "scrf/scrf_util.h"
#include "scrf/make_feat.h"
#include "scrf/scrf_cost.h"
#include <fstream>

struct pruning_env {

    std::ifstream frame_batch;
    std::ifstream ground_truth_batch;
    int min_seg;
    int max_seg;
    scrf::first_order::param_t param;

    std::vector<std::string> features;

    real alpha;
    std::string output;

    std::unordered_map<std::string, int> label_id;
    std::vector<std::string> id_label;
    std::vector<int> labels;
    std::vector<int> id_dim;
    std::vector<int> sils;

    std::unordered_map<std::string, std::string> args;

    pruning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "prune-first",
        "Prune lattice with segmental CRF",
        {
            {"frame-batch", "", true},
            {"gold-batch", "", false},
            {"max-seg", "", false},
            {"min-seg", "", false},
            {"param", "", true},
            {"features", "", true},
            {"alpha", "", true},
            {"output", "", true},
            {"label", "", true},
            {"logprob-label", "", false},
            {"cost-aug", "", false}
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

    if (ebt::in(std::string("gold-batch"), args)) {
        ground_truth_batch.open(args.at("gold-batch"));
    }

    features = ebt::split(args.at("features"), ",");

    alpha = std::stod(args.at("alpha"));
    output = args.at("output");

    label_id = scrf::load_phone_id(args.at("label"));

    id_label.resize(label_id.size());
    for (auto& p: label_id) {
        labels.push_back(p.second);
        id_label[p.second] = p.first;
    }

    sils.push_back(label_id.at("<s>"));
    sils.push_back(label_id.at("</s>"));
    sils.push_back(label_id.at("sil"));

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

        fst::path<scrf::first_order::scrf_t> ground_truth_path;
        scrf::first_order::scrf_t ground_truth;

        if (ebt::in(std::string("cost-aug"), args)) {
            ilat::fst ground_truth_lat = ilat::load_lattice(ground_truth_batch, label_id);

            if (!ground_truth_batch) {
                break;
            }

            ground_truth.fst = std::make_shared<ilat::fst>(ground_truth_lat);
            ground_truth_path = scrf::first_order::make_ground_truth_path(ground_truth);
        }

        std::cout << i << ".lat" << std::endl;

        scrf::first_order::feat_dim_alloc alloc { labels };

        scrf::first_order::composite_feature graph_feat_func
            = scrf::first_order::make_feat(alloc, features, frames, id_dim);

        scrf::first_order::scrf_t graph = scrf::first_order::make_graph_scrf(frames.size(),
            labels, min_seg, max_seg);

        scrf::first_order::composite_weight weight;
        weight.weights.push_back(std::make_shared<scrf::first_order::score::cached_linear_score>(
            scrf::first_order::score::cached_linear_score(param,
                std::make_shared<scrf::first_order::composite_feature>(graph_feat_func))));

        if (ebt::in(std::string("cost-aug"), args)) {
            weight.weights.push_back(std::make_shared<scrf::first_order::cached_seg_cost>(
                scrf::first_order::make_cached_overlap_cost(ground_truth_path, sils)));
        }

        graph.weight_func = std::make_shared<scrf::first_order::composite_weight>(weight);
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

        real sum = 0;
        real max = -inf;

        int edge_count = 0;

        for (auto& e: edges) {
            auto tail = graph.tail(e);
            auto head = graph.head(e);

            int tail_time = graph.fst->time(tail);
            int head_time = graph.fst->time(head);

            real s = fb_alpha(tail) + graph.weight(e) + fb_beta(head);

            if (s > max) {
                max = s;
            }

            if (s != -inf) {
                sum += s;
                ++edge_count;
            }
        }

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

        real threshold = alpha * max + (1 - alpha) * sum / edge_count;

        std::cout << "frames: " << frames.size() << std::endl;
        std::cout << "max: " << max << " avg: " << sum / edge_count
            << " threshold: " << threshold << std::endl;
        std::cout << "forward: " << f_max << " backward: " << b_max << std::endl;

        lattice::fst_data result;

        std::unordered_map<int, int> vertex_map;

        std::vector<int> stack;
        std::unordered_set<int> traversed;

        for (auto v: graph.initials()) {
            stack.push_back(v);
            traversed.insert(v);
        }

        while (stack.size() > 0) {
            auto u = stack.back();
            stack.pop_back();

            for (auto&& e: graph.out_edges(u)) {
                auto tail = graph.tail(e);
                auto head = graph.head(e);

                real weight = graph.weight(e);

                if (fb_alpha(tail) + weight + fb_beta(head) > threshold) {

                    if (!ebt::in(tail, vertex_map)) {
                        int v = vertex_map.size();
                        vertex_map[tail] = v;
                        result.vertices.resize(std::max<int>(result.vertices.size(), v + 1));
                        result.in_edges.resize(std::max<int>(result.vertices.size(), v + 1));
                        result.out_edges.resize(std::max<int>(result.vertices.size(), v + 1));
                        result.in_edges_map.resize(std::max<int>(result.vertices.size(), v + 1));
                        result.out_edges_map.resize(std::max<int>(result.vertices.size(), v + 1));
                        result.vertices.at(v).time = graph.fst->time(tail);
                    }

                    if (!ebt::in(head, vertex_map)) {
                        int v = vertex_map.size();
                        vertex_map[head] = v;
                        result.vertices.resize(std::max<int>(result.vertices.size(), v + 1));
                        result.in_edges.resize(std::max<int>(result.vertices.size(), v + 1));
                        result.out_edges.resize(std::max<int>(result.vertices.size(), v + 1));
                        result.in_edges_map.resize(std::max<int>(result.vertices.size(), v + 1));
                        result.out_edges_map.resize(std::max<int>(result.vertices.size(), v + 1));
                        result.vertices.at(v).time = graph.fst->time(head);
                    }

                    int tail_new = vertex_map.at(tail);
                    int head_new = vertex_map.at(head);
                    int e_new = result.edges.size();

                    result.edges.push_back(lattice::edge_data { tail_new,
                        head_new, weight, id_label[graph.output(e)] });

                    result.in_edges[head_new].push_back(e_new);
                    result.out_edges[tail_new].push_back(e_new);
                    result.in_edges_map[head_new][id_label[graph.output(e)]].push_back(e_new);
                    result.out_edges_map[tail_new][id_label[graph.output(e)]].push_back(e_new);

                    if (!ebt::in(head, traversed)) {
                        stack.push_back(head);
                        traversed.insert(head);
                    }
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
