#include "scrf/util.h"
#include "scrf/scrf.h"
#include "scrf/lm.h"
#include "scrf/lattice.h"
#include <fstream>

struct learning_env {

    std::ifstream input_list;
    std::ifstream lattice_list;
    std::ifstream gold_list;
    std::shared_ptr<lm::fst> lm;
    int max_seg;
    scrf::param_t param;
    scrf::param_t opt_data;
    real step_size;

    std::vector<std::string> features;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

learning_env::learning_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    input_list.open(args.at("input-list"));
    if (ebt::in(std::string("lattice-list"), args)) {
        lattice_list.open(args.at("lattice-list"));
    }
    gold_list.open(args.at("gold-list"));
    lm = std::make_shared<lm::fst>(lm::load_arpa_lm(args.at("lm")));
    max_seg = 20;
    if (ebt::in(std::string("max-seg"), args)) {
        max_seg = std::stoi(args.at("max-seg"));
    }

    param = scrf::load_param(args.at("param"));
    opt_data = scrf::load_param(args.at("opt-data"));
    step_size = std::stod(args.at("step-size"));

    features = ebt::split(args.at("features"), ",");
}

void learning_env::run()
{
    std::string input_file;

    std::shared_ptr<lm::fst> lm_output = scrf::erase_input(lm);

    int i = 0;
    while (std::getline(input_list, input_file)) {

        std::vector<std::vector<real>> inputs = scrf::load_features(input_file);

        std::cout << input_file << std::endl;

        scrf::composite_feature gold_feat_func = scrf::make_feature(features, inputs, max_seg);
        scrf::linear_score gold_score { param, gold_feat_func };

        lattice::fst gold_lat = scrf::load_gold(gold_list);
        scrf::scrf_t gold = scrf::make_gold_scrf(gold_lat, lm);

        scrf::backoff_cost bc; 
        gold.weight_func = std::make_shared<scrf::backoff_cost>(bc);
        gold.topo_order = gold.vertices();
        fst::path<scrf::scrf_t> gold_path = scrf::shortest_path(gold, gold.topo_order);

        gold.weight_func = std::make_shared<scrf::linear_score>(gold_score);
        gold.feature_func = std::make_shared<scrf::composite_feature>(gold_feat_func);

        scrf::composite_feature graph_feat_func = scrf::make_feature(features, inputs, max_seg);
        scrf::linear_score graph_score { param, graph_feat_func };

        scrf::scrf_t graph;
        if (ebt::in(std::string("lattice-list"), args)) {
            lattice::fst lat = lattice::load_lattice(lattice_list);
            lattice::add_eps_loops(lat);

            fst::composed_fst<lattice::fst, lm::fst> comp;
            comp.fst1 = std::make_shared<lattice::fst>(std::move(lat));
            comp.fst2 = lm;
            graph.fst = std::make_shared<decltype(comp)>(comp);

            std::vector<std::tuple<int, int>> topo_order;
            for (auto& v: lattice::topo_order(*(comp.fst1))) {
                for (auto& u: lm->vertices()) {
                    topo_order.push_back(std::make_tuple(v, u));
                }
            }
            graph.topo_order = std::move(topo_order);
        } else {
            graph = scrf::make_graph_scrf(inputs.size(), lm_output, max_seg);
            graph.topo_order = graph.vertices();
        }
        scrf::composite_weight cost_aug_weight;
        cost_aug_weight.weights.push_back(std::make_shared<scrf::linear_score>(graph_score));
        cost_aug_weight.weights.push_back(std::make_shared<scrf::overlap_cost>(scrf::overlap_cost { gold_path }));
        graph.weight_func = std::make_shared<scrf::composite_weight>(cost_aug_weight);
        graph.feature_func = std::make_shared<scrf::composite_feature>(graph_feat_func);

        scrf::hinge_loss loss_func { gold_path, graph };

        real ell = loss_func.loss();

        std::cout << "loss: " << ell << std::endl;

        if (ell < 0) {
            std::cout << "loss is less than zero.  skipping." << std::endl;
            // exit(1);
        }

        std::cout << std::endl;

        if (ell > 0) {
            auto param_grad = loss_func.param_grad();

#if 0
            {
                real& f = param.class_param.at("<s>").at(0);
                real tmp = f;
                f += 1e-8;
                scrf::hinge_loss loss_func_step { gold_path, graph };
                std::cout << param_grad.class_param.at("<s>").at(0)
                    << " " << (loss_func_step.loss() - ell) / 1e-8 << std::endl;
                f = tmp;
            }
#endif

            scrf::adagrad_update(param, param_grad, opt_data, step_size);

            scrf::save_param("param-last", param);
            scrf::save_param("opt-data-last", opt_data);
        }

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
            {"lattice-list", "", false},
            {"gold-list", "", true},
            {"lm", "", true},
            {"max-seg", "", false},
            {"param", "", true},
            {"opt-data", "", true},
            {"step-size", "", true},
            {"features", "", true}
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
