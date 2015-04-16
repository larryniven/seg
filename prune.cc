#include "scrf/util.h"
#include "scrf/scrf.h"
#include "scrf/lm.h"
#include "scrf/lattice.h"
#include <fstream>

struct pruning_env {

    std::ifstream input_list;
    std::shared_ptr<lm::fst> lm;
    int max_seg;
    scrf::param_t param;

    std::vector<std::string> features;

    real alpha;
    std::string output;

    std::unordered_map<std::string, std::string> args;

    pruning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

pruning_env::pruning_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    input_list.open(args.at("input-list"));
    lm = std::make_shared<lm::fst>(lm::load_arpa_lm(args.at("lm")));
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
    std::ofstream ofs { output };

    std::string input_file;

    std::shared_ptr<lm::fst> lm_output = scrf::erase_input(lm);

    int i = 0;
    while (std::getline(input_list, input_file)) {

        std::vector<std::vector<real>> inputs = scrf::load_features(input_file);

        std::cout << input_file << std::endl;

        scrf::composite_feature graph_feat_func = scrf::make_feature(features, inputs, max_seg);
        scrf::linear_score graph_score { param, graph_feat_func };

        scrf::scrf_t graph = scrf::make_graph_scrf(inputs.size(), lm_output, max_seg);
        graph.topo_order = graph.vertices();
        graph.weight_func = std::make_shared<scrf::linear_score>(graph_score);
        graph.feature_func = std::make_shared<scrf::composite_feature>(graph_feat_func);

        auto edges = graph.edges();

        auto order = graph.topo_order;

        fst::forward_one_best<scrf::scrf_t> forward;
        for (auto v: graph.initials()) {
            forward.extra[v] = {std::make_tuple(-1, -1), 0};
        }
        forward.merge(graph, order);

        std::reverse(order.begin(), order.end());
        fst::backward_one_best<scrf::scrf_t> backward;
        for (auto v: graph.finals()) {
            backward.extra[v] = {std::make_tuple(-1, -1), 0};
        }
        backward.merge(graph, order);

        auto fb_alpha = [&](std::tuple<int, int> const& v) {
            return forward.extra[v].value;
        };

        auto fb_beta = [&](std::tuple<int, int> const& v) {
            return backward.extra[v].value;
        };

        real inf = std::numeric_limits<real>::infinity();

        real sum = 0;
        real max = -inf;

        for (auto& e: edges) {
            auto tail = graph.tail(e);
            auto head = graph.head(e);

            int tail_time = graph.fst->fst1->data->vertices.at(std::get<0>(tail)).time;
            int head_time = graph.fst->fst1->data->vertices.at(std::get<0>(head)).time;

            real s = fb_alpha(tail) + graph.weight(e) + fb_beta(head);

            if (s > max) {
                max = s;
            }

            if (s != -inf) {
                sum += s;
            }
        }

        real threshold = alpha * max + (1 - alpha) * sum / edges.size();

        std::cout << "frames: " << inputs.size() << std::endl;
        std::cout << "max: " << max << " avg: " << sum / edges.size()
            << " threshold: " << threshold << std::endl;

        lattice::fst_data result;

        std::unordered_map<std::tuple<int, int>, int> vertex_map;

        auto time = [&](std::tuple<int, int> const& v) {
            auto& lat = *(graph.fst->fst1);

            return lat.data->vertices.at(std::get<0>(v)).time;
        };

        std::vector<std::tuple<int, int>> stack;
        std::unordered_set<std::tuple<int, int>> traversed;

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
                        result.vertices.at(v).time = time(tail);
                    }

                    if (!ebt::in(head, vertex_map)) {
                        int v = vertex_map.size();
                        vertex_map[head] = v;
                        result.vertices.resize(std::max<int>(result.vertices.size(), v + 1));
                        result.in_edges.resize(std::max<int>(result.vertices.size(), v + 1));
                        result.out_edges.resize(std::max<int>(result.vertices.size(), v + 1));
                        result.in_edges_map.resize(std::max<int>(result.vertices.size(), v + 1));
                        result.out_edges_map.resize(std::max<int>(result.vertices.size(), v + 1));
                        result.vertices.at(v).time = time(head);
                    }

                    int tail_new = vertex_map.at(tail);
                    int head_new = vertex_map.at(head);
                    int e_new = result.edges.size();

                    result.edges.push_back(lattice::edge_data { graph.output(e), tail_new,
                        head_new, weight });

                    result.in_edges[head_new].push_back(e_new);
                    result.out_edges[tail_new].push_back(e_new);
                    result.in_edges_map[head_new][graph.output(e)].push_back(e_new);
                    result.out_edges_map[tail_new][graph.output(e)].push_back(e_new);

                    if (!ebt::in(head, traversed)) {
                        stack.push_back(head);
                        traversed.insert(head);
                    }
                }
            }
        }

        lattice::fst result_fst;
        result_fst.data = std::make_shared<lattice::fst_data>(result);

        ofs << input_file << std::endl;

        for (int e = 0; e < result.edges.size(); ++e) {
            int tail = result_fst.tail(e);
            int head = result_fst.head(e);

            ofs << int(result_fst.data->vertices.at(tail).time * 1e5)
                << " " << int(result_fst.data->vertices.at(head).time *1e5)
                << " " << result_fst.output(e)
                << " " << tail << " " << head
                << " " << result.edges.at(e).weight << std::endl;
        }
        ofs << "." << std::endl;

        std::cout << "edges: " << edges.size() << " left: " << result.edges.size()
            << " (" << real(result.edges.size()) / edges.size() << ")" << std::endl;

        std::cout << std::endl;

        ++i;

#if 0
        if (i == 10) {
            exit(1);
        }
#endif

    }
}

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "learn",
        "Learn segmental CRF",
        {
            {"input-list", "", true},
            {"lm", "", true},
            {"max-seg", "", false},
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
