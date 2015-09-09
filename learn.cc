#include "scrf/util.h"
#include "scrf/scrf.h"
#include "scrf/lm.h"
#include "scrf/lattice.h"
#include "speech/speech.h"
#include "scrf/nn.h"
#include "scrf/weiran.h"
#include "scrf/loss.h"
#include "scrf/cost.h"
#include "scrf/make_feature.h"
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

    std::vector<real> cm_mean;
    std::vector<real> cm_stddev;
    nn::param_t nn_param;
    nn::param_t nn_opt_data;
    nn::nn_t nn;
    real nn_step_size;

    int save_every;
    int beam_width;

    std::string output_param;
    std::string output_opt_data;

    std::vector<std::string> features;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

learning_env::learning_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    if (ebt::in(std::string("input-list"), args)) {
        input_list.open(args.at("input-list"));
    }

    if (ebt::in(std::string("lattice-list"), args)) {
        lattice_list.open(args.at("lattice-list"));
    }

    gold_list.open(args.at("gold-list"));

    lm = std::make_shared<lm::fst>(lm::load_arpa_lm(args.at("lm")));

    max_seg = 20;
    if (ebt::in(std::string("max-seg"), args)) {
        max_seg = std::stoi(args.at("max-seg"));
    }

    if (ebt::in(std::string("cm-mean"), args)) {
        std::ifstream ifs { args.at("cm-mean") };
        std::string line;
        std::getline(ifs, line);
        std::vector<std::string> parts = ebt::split(line);
        for (auto& v: parts) {
            cm_mean.push_back(std::stod(v));
        }
    }

    if (ebt::in(std::string("cm-stddev"), args)) {
        std::ifstream ifs { args.at("cm-stddev") };
        std::string line;
        std::getline(ifs, line);
        std::vector<std::string> parts = ebt::split(line);
        for (auto& v: parts) {
            cm_stddev.push_back(std::stod(v));
        }
    }

    if (ebt::in(std::string("nn-param"), args)) {
        nn_param = nn::load_param(args.at("nn-param"));
        nn = nn::make_nn(nn_param);
    }

    if (ebt::in(std::string("weiran-nn-param"), args)) {
        nn_param = nn::load_param(args.at("weiran-nn-param"));
        nn = weiran::make_nn(nn_param);
    }

    if (ebt::in(std::string("nn-opt-data"), args)) {
        nn_opt_data = nn::load_param(args.at("nn-opt-data"));
    }

    if (ebt::in(std::string("nn-step-size"), args)) {
        nn_step_size = std::stod(args.at("nn-step-size"));
    }

    param = scrf::load_param(args.at("param"));
    opt_data = scrf::load_param(args.at("opt-data"));
    step_size = std::stod(args.at("step-size"));

    if (ebt::in(std::string("beam-width"), args)) {
        beam_width = std::stoi(args.at("beam-width"));
    }

    if (ebt::in(std::string("save-every"), args)) {
        save_every = std::stoi(args.at("save-every"));
    } else {
        save_every = std::numeric_limits<int>::max();
    }

    features = ebt::split(args.at("features"), ",");

    output_param = "param-last";
    if (ebt::in(std::string("output-param"), args)) {
        output_param = args.at("output-param");
    }

    output_opt_data = "opt-data-last";
    if (ebt::in(std::string("output-opt-data"), args)) {
        output_opt_data = args.at("output-opt-data");
    }
}

void learning_env::run()
{
    std::string input_file;

    std::shared_ptr<lm::fst> lm_output = scrf::erase_input(lm);

    int i = 1;
    while (1) {

        std::vector<std::vector<real>> inputs;

        if (std::getline(input_list, input_file)) {
            inputs = speech::load_frames(input_file);
        }

        lattice::fst gold_lat = lattice::load_lattice(gold_list);

        if (!gold_list) {
            break;
        }

        std::cout << input_file << std::endl;

        scrf::composite_feature gold_feat_func = scrf::make_feature(features, inputs, max_seg,
            cm_mean, cm_stddev, nn);

        scrf::scrf_t gold = scrf::make_gold_scrf(gold_lat, lm);

        scrf::backoff_cost bc; 
        gold.weight_func = std::make_shared<scrf::backoff_cost>(bc);
        gold.topo_order = topo_order(gold);
        fst::path<scrf::scrf_t> gold_path = scrf::shortest_path(gold, gold.topo_order);

        gold.weight_func = std::make_shared<scrf::composite_weight>(
            scrf::make_weight(param, gold_feat_func));
        gold.feature_func = std::make_shared<scrf::composite_feature>(gold_feat_func);

        scrf::composite_feature graph_feat_func = scrf::make_feature(features, inputs, max_seg,
            cm_mean, cm_stddev, nn);

        scrf::scrf_t graph;
        if (ebt::in(std::string("lattice-list"), args)) {
            lattice::fst lat = lattice::load_lattice(lattice_list);

            if (!lattice_list) {
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
            graph = scrf::make_graph_scrf(inputs.size(), lm_output, max_seg);
            graph.topo_order = scrf::topo_order(graph);
        }
        scrf::composite_weight cost_aug_weight;
        cost_aug_weight.weights.push_back(std::make_shared<scrf::composite_weight>(
            scrf::make_weight(param, graph_feat_func)));
        cost_aug_weight.weights.push_back(std::make_shared<scrf::overlap_cost>(
            scrf::overlap_cost { gold_path }));
        graph.weight_func = std::make_shared<scrf::composite_weight>(cost_aug_weight);
        graph.feature_func = std::make_shared<scrf::composite_feature>(graph_feat_func);

        std::shared_ptr<scrf::loss_func> loss_func;
        if (args.at("loss") == "hinge") {
            loss_func = std::make_shared<scrf::hinge_loss>(scrf::hinge_loss { gold_path, graph });
        } else if (args.at("loss") == "hinge-beam") {
            loss_func = std::make_shared<scrf::hinge_loss_beam>(scrf::hinge_loss_beam { gold_path, graph, beam_width });
        } else if (args.at("loss") == "filtering") {
            real alpha = std::stod(args.at("alpha"));
            loss_func = std::make_shared<scrf::filtering_loss>(scrf::filtering_loss { gold_path, graph, alpha });
        } else {
            std::cout << "unknown loss function " << args.at("loss") << std::endl;
            exit(1);
        }

        real ell = loss_func->loss();

        std::cout << "loss: " << ell << std::endl;

        if (ell < 0) {
            std::cout << "loss is less than zero.  skipping." << std::endl;
        }

        std::cout << std::endl;

        if (ell > 0) {
            auto param_grad = loss_func->param_grad();
            if (args.at("loss") == "hinge" && ebt::in(std::string("backprop"), args)) {
                scrf::hinge_loss& hinge_loss = static_cast<scrf::hinge_loss&>(*loss_func);
                nn::param_t nn_grad = nn::hinge_nn_grad(
                    nn, param, hinge_loss.gold, hinge_loss.graph_path,
                    graph_feat_func);
                nn::move_out_param(nn, nn_param);
                nn::adagrad_update(nn_param, nn_grad, nn_opt_data, nn_step_size);
                if (i % save_every == 0) {
                    nn::save_param(nn_param, "nn-param-last");
                    nn::save_param(nn_opt_data, "nn-opt-data-last");
                }
                nn::move_in_param(nn, nn_param);
            }
            scrf::adagrad_update(param, param_grad, opt_data, step_size);

            if (i % save_every == 0) {
                scrf::save_param("param-last", param);
                scrf::save_param("opt-data-last", opt_data);
            }
        }

#if DEBUG_TOP_10
        if (i == 10) {
            exit(1);
        }
#endif

        ++i;
    }

    scrf::save_param(output_param, param);
    scrf::save_param(output_opt_data, opt_data);

    if (args.at("loss") == "hinge" && ebt::in(std::string("backprop"), args)) {
        nn::move_out_param(nn, nn_param);
        nn::save_param(nn_param, "nn-param-last");
        nn::save_param(nn_opt_data, "nn-opt-data-last");
    }
}

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "learn",
        "Learn segmental CRF",
        {
            {"input-list", "", false},
            {"lattice-list", "", false},
            {"gold-list", "", true},
            {"lm", "", true},
            {"max-seg", "", false},
            {"param", "", true},
            {"opt-data", "", true},
            {"step-size", "", true},
            {"features", "", true},
            {"alpha", "filtering parameter", false},
            {"loss", "hinge,filtering", true},
            {"cm-mean", "", false},
            {"cm-stddev", "", false},
            {"weiran-nn-param", "", false},
            {"nn-param", "", false},
            {"nn-opt-data", "", false},
            {"nn-step-size", "", false},
            {"backprop", "", false},
            {"beam-width", "", false},
            {"save-every", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false}
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
