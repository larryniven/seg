#include "scrf/util.h"
#include "scrf/scrf.h"
#include "opt/opt.h"
#include <fstream>

struct learning_env {

    std::ifstream feature_list;
    std::ifstream gold_list;
    int max_seg;
    scrf::detail::model_vector weights;
    scrf::detail::model_vector opt_data;

    std::unordered_map<std::string, int> phone_map;
    std::vector<std::string> inv_phone_map;

    real step_size;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();
};

learning_env::learning_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    feature_list.open(args.at("feature-list"));
    gold_list.open(args.at("gold-list"));
    max_seg = 10;
    if (ebt::in(std::string("max-seg"), args)) {
        max_seg = std::stoi(args.at("max-seg"));
    }

    weights = scrf::detail::load_model(args.at("model"));
    opt_data = scrf::detail::load_model(args.at("opt-data"));

    phone_map = scrf::detail::load_phone_map(args.at("phone-set"));
    inv_phone_map = scrf::detail::make_inv_phone_map(phone_map);

    step_size = std::stod(args.at("step-size"));
}

void learning_env::run()
{
    std::string feature_file;

    int i = 0;
    while (std::getline(feature_list, feature_file)) {

        std::vector<std::vector<real>> features = scrf::load_features(feature_file);

        scrf::detail::gold_scrf gold { features, weights, int(phone_map.size()) };
        gold.edges = scrf::detail::load_gold(gold_list, phone_map, int(features.size()));

        scrf::detail::graph_scrf graph { features, weights, int(phone_map.size()),
            phone_map, int(features.size()), max_seg };

        scrf::detail::log_loss loss { gold, graph };

        real ell = loss.loss();

        std::cout << i << ": " << "loss: " << ell << std::endl;

        auto grad = loss.model_grad();

#if 0
        {
            real& f = weights.class_weights.at(0).at(0);
            real tmp = f;
            f += 1e-8;
            scrf::detail::graph_scrf graph_step { features, weights, int(phone_map.size()),
                phone_map, int(features.size()), max_seg };
            scrf::detail::log_loss loss_step { gold, graph_step };
            std::cout << "grad check: " << (loss_step.loss() - ell) / 1e-8 << " " << grad.at(0).at(0) << std::endl;
            f = tmp;
        }

        {
            real& f = weights.class_weights.at(0).at(1);
            real tmp = f;
            f += 1e-8;
            scrf::detail::graph_scrf graph_step { features, weights, int(phone_map.size()),
                phone_map, int(features.size()), max_seg };
            scrf::detail::log_loss loss_step { gold, graph_step };
            std::cout << "grad check: " << (loss_step.loss() - ell) / 1e-8 << " " << grad.at(0).at(1) << std::endl;
            f = tmp;
        }

        {
            real& f = weights.class_weights.at(1).at(0);
            real tmp = f;
            f += 1e-8;
            scrf::detail::graph_scrf graph_step { features, weights, int(phone_map.size()),
                phone_map, int(features.size()), max_seg };
            scrf::detail::log_loss loss_step { gold, graph_step };
            std::cout << "grad check: " << (loss_step.loss() - ell) / 1e-8 << " " << grad.at(1).at(0) << std::endl;
            f = tmp;
        }

        {
            real& f = weights.class_weights.at(1).at(1);
            real tmp = f;
            f += 1e-8;
            scrf::detail::graph_scrf graph_step { features, weights, int(phone_map.size()),
                phone_map, int(features.size()), max_seg };
            scrf::detail::log_loss loss_step { gold, graph_step };
            std::cout << "grad check: " << (loss_step.loss() - ell) / 1e-8 << " " << grad.at(1).at(1) << std::endl;
            f = tmp;
        }

#endif

        opt::adagrad_update(weights.class_weights, grad, opt_data.class_weights, step_size);

        ++i;

        save_model("model-last", weights);
        save_model("opt-data-last", opt_data);

        if (i == 10) {
            exit(1);
        }

    }

}

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "learn",
        "Learn segmental CRF",
        {
            {"feature-list", "", true},
            {"gold-list", "", true},
            {"max-seg", "", false},
            {"model", "", true},
            {"opt-data", "", true},
            {"phone-set", "", true},
            {"step-size", "", true}
        }
    };

    if (argc == 1) {
        ebt::usage(spec);
        exit(1);
    }

    auto args = ebt::parse_args(argc, argv, spec);

    std::cout << args << std::endl;

    learning_env env { args };

    env.run();

    return 0;
}
