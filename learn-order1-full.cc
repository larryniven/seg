#include "scrf/fscrf.h"
#include "scrf/loss.h"
#include "scrf/scrf_weight.h"
#include "scrf/util.h"
#include <fstream>

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream gt_batch;

    int save_every;

    std::string output_param;
    std::string output_opt_data;

    fscrf::learning_args l_args;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "learn-first",
        "Learn segmental CRF",
        {
            {"frame-batch", "", false},
            {"gt-batch", "", true},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"param", "", true},
            {"opt-data", "", true},
            {"step-size", "", true},
            {"decay", "", false},
            {"momentum", "", false},
            {"features", "", true},
            {"save-every", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"loss", "", true},
            {"cost-scale", "", false},
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

    learning_env env { args };

    env.run();

    return 0;
}

learning_env::learning_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    if (ebt::in(std::string("frame-batch"), args)) {
        frame_batch.open(args.at("frame-batch"));
    }

    gt_batch.open(args.at("gt-batch"));

    save_every = std::numeric_limits<int>::max();
    if (ebt::in(std::string("save-every"), args)) {
        save_every = std::stoi(args.at("save-every"));
    }

    output_param = "param-last";
    if (ebt::in(std::string("output-param"), args)) {
        output_param = args.at("output-param");
    }

    output_opt_data = "opt-data-last";
    if (ebt::in(std::string("output-opt-data"), args)) {
        output_opt_data = args.at("output-opt-data");
    }

    fscrf::parse_learning_args(l_args, args);
}

void learning_env::run()
{
    ebt::Timer timer;

    int i = 1;

    while (1) {

        fscrf::learning_sample s { l_args };

        s.frames = speech::load_frame_batch(frame_batch);

        s.gt_segs = util::load_segments(gt_batch, l_args.label_id);

        if (!gt_batch) {
            break;
        }

        fscrf::make_graph(s, l_args);

        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree
            = tensor_tree::make_var_tree(comp_graph, l_args.param);

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        for (int i = 0; i < s.frames.size(); ++i) {
            frame_ops.push_back(comp_graph.var(la::vector<double>(s.frames[i])));
        }

        auto frame_mat = autodiff::col_cat(frame_ops);

        s.graph_data.weight_func = fscrf::make_weights(l_args.features, var_tree, frame_mat);

        fscrf::hinge_loss loss_func { s.graph_data, s.gt_segs, l_args.sils, l_args.cost_scale };

        double ell = loss_func.loss();

        std::cout << "loss: " << ell << std::endl;

#if 0
        {
            fscrf::learning_args l_args2 = l_args;
            l_args2.param = tensor_tree::copy_tree(l_args.param);
            l_args2.opt_data = tensor_tree::copy_tree(l_args.opt_data);

            auto& m = tensor_tree::get_matrix(l_args2.param->children[2]);
            m(l_args.label_id.at("sil") - 1, 0) += 1e-8;

            fscrf::learning_sample s2 { l_args2 };
            s2.frames = s.frames;
            s2.gt_segs = s.gt_segs;

            fscrf::make_graph(s2, l_args);

            autodiff::computation_graph comp_graph2;
            std::shared_ptr<tensor_tree::vertex> var_tree2
                = tensor_tree::make_var_tree(comp_graph2, l_args2.param);

            std::vector<std::shared_ptr<autodiff::op_t>> frame_ops2;
            for (int i = 0; i < s2.frames.size(); ++i) {
                frame_ops2.push_back(comp_graph2.var(la::vector<double>(s2.frames[i])));
            }

            auto frame_mat2 = autodiff::col_cat(frame_ops2);

            s2.graph_data.weight_func = fscrf::make_weights(l_args2.features, var_tree2, frame_mat2);

            fscrf::hinge_loss loss_func2 { s2.graph_data, s2.gt_segs, l_args2.sils, l_args2.cost_scale };

            double ell2 = loss_func2.loss();

            std::cout << "numeric grad: " << (ell2 - ell) / 1e-8 << std::endl;

        }
#endif

        std::shared_ptr<tensor_tree::vertex> param_grad = fscrf::make_tensor_tree(l_args.features);

        if (ell > 0) {
            loss_func.grad();

            s.graph_data.weight_func->grad();

            tensor_tree::copy_grad(param_grad, var_tree);

            auto& m = tensor_tree::get_matrix(param_grad->children[2]);

            std::cout << "analytic grad: " << m(l_args.label_id.at("sil") - 1, 0) << std::endl;

            double v1 = tensor_tree::get_matrix(l_args.param->children[0])(l_args.label_id.at("sil") - 1, 0);

            if (ebt::in(std::string("decay"), l_args.args)) {
                tensor_tree::rmsprop_update(l_args.param, param_grad, l_args.opt_data,
                    l_args.decay, l_args.step_size);
            } else {
                tensor_tree::adagrad_update(l_args.param, param_grad, l_args.opt_data,
                    l_args.step_size);
            }

            double v2 = tensor_tree::get_matrix(l_args.param->children[0])(l_args.label_id.at("sil") - 1, 0);

            std::cout << "weight: " << v1 << " update: " << v2 - v1 << " ratio: " << (v2 - v1) / v1 << std::endl;
        }

        if (ell < 0) {
            std::cout << "loss is less than zero.  skipping." << std::endl;
        }

        std::cout << "gold segs: " << s.gt_segs.size()
            << " frames: " << s.frames.size() << std::endl;

        std::cout << std::endl;

        if (i % save_every == 0) {
            tensor_tree::save_tensor(l_args.param, "param-last");
            tensor_tree::save_tensor(l_args.opt_data, "opt-data-last");
        }

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        ++i;
    }

    tensor_tree::save_tensor(l_args.param, output_param);
    tensor_tree::save_tensor(l_args.opt_data, output_opt_data);

}

