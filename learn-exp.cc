#include "scrf/iscrf.h"
#include "scrf/loss.h"
#include "scrf/scrf_weight.h"
#include <fstream>

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream ground_truth_batch;

    int save_every;

    std::string output_param;
    std::string output_opt_data;

    scrf::experimental::learning_args l_args;

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
            {"frame-batch", "", true},
            {"ground-truth-batch", "", true},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"min-cost-path", "Use min cost path for training", false},
            {"param", "", true},
            {"opt-data", "", true},
            {"step-size", "", true},
            {"momentum", "", false},
            {"features", "", true},
            {"save-every", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"loss", "", true},
            {"label", "", true},
            {"label-dim", "", false},
            {"length-stat", "", false},
            {"alpha", "", false},
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

    ground_truth_batch.open(args.at("ground-truth-batch"));

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

    l_args = scrf::experimental::parse_learning_args(args);
}

void learning_env::run()
{
    int i = 1;

    while (1) {

        scrf::experimental::learning_sample s { l_args };

        s.frames = speech::load_frame_batch(frame_batch);

        s.ground_truth_fst = ilat::load_lattice(ground_truth_batch, l_args.label_id);

        if (!ground_truth_batch) {
            break;
        }

        std::cout << s.ground_truth_fst.data->name << std::endl;

        std::cout << "ground truth: ";
        for (auto& e: s.ground_truth_fst.edges()) {
            std::cout << l_args.id_label[s.ground_truth_fst.output(e)] << " ";
        }
        std::cout << std::endl;

        if (ebt::in(std::string("min-cost-path"), args)) {
            scrf::experimental::make_min_cost_gold(s, l_args);
        } else {
            scrf::experimental::make_gold(s, l_args);
        }

        s.cost = std::make_shared<scrf::experimental::seg_cost<ilat::fst>>(
            scrf::experimental::make_overlap_cost<ilat::fst>(s.ground_truth.fst, l_args.sils));

        double gold_cost = 0;
    
        std::cout << "gold path: ";
        for (auto& e: s.gold.edges()) {
            std::cout << l_args.id_label[s.gold.output(e)] << " ";
            gold_cost += (*s.cost)(s.gold.fst, e);
        }
        std::cout << std::endl;
    
        std::cout << "gold cost: " << gold_cost << std::endl;

        scrf::experimental::make_graph(s, l_args);

        std::shared_ptr<scrf::experimental::loss_func<scrf::experimental::dense_vec>> loss_func;

        if (args.at("loss") == "hinge-loss") {
            scrf::experimental::composite_weight<ilat::fst>& graph_weight_func
                = *dynamic_cast<scrf::experimental::composite_weight<ilat::fst>*>(s.graph.weight_func.get());
            graph_weight_func.weights.push_back(s.cost);

            using hinge_loss = scrf::experimental::hinge_loss<
                scrf::experimental::iscrf, scrf::experimental::dense_vec,
                scrf::experimental::iscrf_path_maker>;

            loss_func = std::make_shared<hinge_loss>(hinge_loss { s.gold, s.graph });

            hinge_loss const& loss = *dynamic_cast<hinge_loss*>(loss_func.get());

            double gold_weight = 0;

            std::cout << "gold: ";
            for (auto& e: s.gold.edges()) {
                std::cout << l_args.id_label[s.gold.output(e)] << " ";
                gold_weight += s.gold.weight(e);
            }
            std::cout << std::endl;

            std::cout << "gold score: " << gold_weight << std::endl;

            double graph_weight = 0;

            std::cout << "cost aug: ";
            for (auto& e: loss.graph_path.edges()) {
                std::cout << l_args.id_label[loss.graph.output(e)] << " ";
                graph_weight += loss.graph_path.weight(e);
            }
            std::cout << std::endl;

            std::cout << "cost aug score: " << graph_weight << std::endl; 
        } else {
            std::cout << "unknown loss function " << args.at("loss") << std::endl;
            exit(1);
        }

        std::cout << "gold segs: " << s.gold.edges().size()
            << " frames: " << s.frames.size() << std::endl;

        double ell = loss_func->loss();

        std::cout << "loss: " << ell << std::endl;

        scrf::experimental::dense_vec param_grad;

        if (ell > 0) {
            param_grad = loss_func->param_grad();
        }

        if (ell < 0) {
            std::cout << "loss is less than zero.  skipping." << std::endl;
        }

        std::cout << std::endl;

        scrf::experimental::adagrad_update(l_args.param, param_grad, l_args.opt_data, l_args.step_size);

        if (i % save_every == 0) {
            scrf::experimental::save_vec(l_args.param, "param-last");
            scrf::experimental::save_vec(l_args.opt_data, "opt-data-last");
        }

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        ++i;
    }

    scrf::experimental::save_vec(l_args.param, output_param);
    scrf::experimental::save_vec(l_args.opt_data, output_opt_data);

}

