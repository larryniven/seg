#include "scrf/util.h"
#include "scrf/scrf.h"
#include "scrf/lm.h"
#include "scrf/lattice.h"
#include "speech/speech.h"
#include <fstream>

struct pruning_env {

    std::ifstream input_list;
    std::ifstream lattice_list;
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
    if (ebt::in(std::string("lattice-list"), args)) {
        lattice_list.open(args.at("lattice-list"));
    }
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

        std::vector<std::vector<real>> inputs = speech::load_frames(input_file);

        std::cout << input_file << std::endl;

        scrf::composite_feature graph_feat_func = scrf::make_feature(features, inputs, max_seg);

        scrf::scrf_t graph;

        if (ebt::in(std::string("lattice-list"), args)) {
            lattice::fst lat = lattice::load_lattice(lattice_list);
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
            graph = scrf::make_graph_scrf(inputs.size(), lm_output, max_seg);
            graph.topo_order = graph.vertices();
        }
        graph.weight_func = std::make_shared<scrf::composite_weight>(
            scrf::make_weight(param, graph_feat_func));
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

        real inf = std::numeric_limits<real>::infinity();

        auto fb_alpha = [&](std::tuple<int, int> const& v) {
            if (ebt::in(v, forward.extra)) {
                return forward.extra[v].value;
            } else {
                return -inf;
            }
        };

        auto fb_beta = [&](std::tuple<int, int> const& v) {
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

            int tail_time = graph.fst->fst1->data->vertices.at(std::get<0>(tail)).time;
            int head_time = graph.fst->fst1->data->vertices.at(std::get<0>(head)).time;

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

        std::cout << "frames: " << inputs.size() << std::endl;
        std::cout << "max: " << max << " avg: " << sum / edge_count
            << " threshold: " << threshold << std::endl;
        std::cout << "forward: " << f_max << " backward: " << b_max << std::endl;

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

#if DEBUG_TOP_10
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
            {"lattice-list", "", false},
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
