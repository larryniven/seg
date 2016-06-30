#include "scrf/fscrf.h"
#include "scrf/loss.h"
#include "scrf/scrf_weight.h"
#include "scrf/util.h"
#include <fstream>

struct prediction_env {

    std::ifstream frame_batch;

    fscrf::inference_args i_args;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "learn-first",
        "Learn segmental CRF",
        {
            {"frame-batch", "", false},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"param", "", true},
            {"features", "", true},
            {"label", "", true},
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

    fscrf::parse_inference_args(i_args, args);
}

void prediction_env::run()
{
    int i = 1;

    while (1) {

        fscrf::sample s { i_args };

        s.frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        fscrf::make_graph(s, i_args);

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, i_args.param);

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        for (int i = 0; i < s.frames.size(); ++i) {
            frame_ops.push_back(comp_graph.var(la::vector<double>(s.frames[i])));
        }

        auto frame_mat = autodiff::col_cat(frame_ops);

        scrf::composite_weight<ilat::fst> weight_func;
        weight_func.weights.push_back(std::make_shared<fscrf::frame_avg_score>(
            fscrf::frame_avg_score(tensor_tree::get_var(var_tree->children[0]), frame_mat)));
        weight_func.weights.push_back(std::make_shared<fscrf::frame_samples_score>(
            fscrf::frame_samples_score(tensor_tree::get_var(var_tree->children[1]), frame_mat, 1.0 / 6)));
        weight_func.weights.push_back(std::make_shared<fscrf::frame_samples_score>(
            fscrf::frame_samples_score(tensor_tree::get_var(var_tree->children[2]), frame_mat, 1.0 / 2)));
        weight_func.weights.push_back(std::make_shared<fscrf::frame_samples_score>(
            fscrf::frame_samples_score(tensor_tree::get_var(var_tree->children[3]), frame_mat, 5.0 / 6)));
        weight_func.weights.push_back(std::make_shared<fscrf::left_boundary_score>(
            fscrf::left_boundary_score(tensor_tree::get_var(var_tree->children[4]), frame_mat, -1)));
        weight_func.weights.push_back(std::make_shared<fscrf::left_boundary_score>(
            fscrf::left_boundary_score(tensor_tree::get_var(var_tree->children[5]), frame_mat, -2)));
        weight_func.weights.push_back(std::make_shared<fscrf::left_boundary_score>(
            fscrf::left_boundary_score(tensor_tree::get_var(var_tree->children[6]), frame_mat, -3)));
        weight_func.weights.push_back(std::make_shared<fscrf::right_boundary_score>(
            fscrf::right_boundary_score(tensor_tree::get_var(var_tree->children[7]), frame_mat, 1)));
        weight_func.weights.push_back(std::make_shared<fscrf::right_boundary_score>(
            fscrf::right_boundary_score(tensor_tree::get_var(var_tree->children[8]), frame_mat, 2)));
        weight_func.weights.push_back(std::make_shared<fscrf::right_boundary_score>(
            fscrf::right_boundary_score(tensor_tree::get_var(var_tree->children[9]), frame_mat, 3)));
        weight_func.weights.push_back(std::make_shared<fscrf::length_score>(
            fscrf::length_score { tensor_tree::get_var(var_tree->children[10]) }));
        weight_func.weights.push_back(std::make_shared<fscrf::log_length_score>(
            fscrf::log_length_score { tensor_tree::get_var(var_tree->children[11]) }));
        weight_func.weights.push_back(std::make_shared<fscrf::bias_score>(
            fscrf::bias_score { tensor_tree::get_var(var_tree->children[12]) }));
        s.graph_data.weight_func = std::make_shared<scrf::composite_weight<ilat::fst>>(weight_func);

        fscrf::fscrf_data graph_path_data;
        graph_path_data.fst = scrf::shortest_path(s.graph_data);

        fscrf::fscrf_fst graph_path { graph_path_data };

        for (auto& e: graph_path.edges()) {
            std::cout << i_args.id_label.at(graph_path.output(e)) << " ";
        }
        std::cout << "(" << i << ".dot)" << std::endl;

        ++i;
    }
}

