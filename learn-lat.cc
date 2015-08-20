#include "scrf/util.h"
#include "scrf/scrf.h"
#include "scrf/lm.h"
#include "scrf/lattice.h"
#include "speech/speech.h"
#include "scrf/nn.h"
#include "scrf/loss.h"
#include "scrf/cost.h"
#include "scrf/make_feature.h"
#include <fstream>

struct learning_env {

    std::ifstream lat_list;
    std::ifstream gold_list;
    std::shared_ptr<lm::fst> lm;
    scrf::param_t param;
    scrf::param_t opt_data;
    real step_size;

    int save_every;

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
    lat_list.open(args.at("lat-list"));
    gold_list.open(args.at("gold-list"));

    lm = std::make_shared<lm::fst>(lm::load_arpa_lm(args.at("lm")));

    param = scrf::load_param(args.at("param"));
    opt_data = scrf::load_param(args.at("opt-data"));
    step_size = std::stod(args.at("step-size"));

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
    int i = 1;
    while (1) {

        lattice::fst gold_lat = lattice::load_lattice(gold_list);

        if (!gold_list) {
            break;
        }

        scrf::composite_feature gold_feat_func = scrf::make_feature3(features);

        scrf::scrf_t gold = scrf::make_gold_scrf(gold_lat, lm);

        scrf::backoff_cost bc; 
        gold.weight_func = std::make_shared<scrf::backoff_cost>(bc);
        gold.topo_order = topo_order(gold);
        fst::path<scrf::scrf_t> gold_path = scrf::shortest_path(gold, gold.topo_order);

        gold.weight_func = std::make_shared<scrf::linear_weight>(
            scrf::linear_weight { param, gold_feat_func });
        gold.feature_func = std::make_shared<scrf::composite_feature>(gold_feat_func);

        scrf::composite_feature graph_feat_func = scrf::make_feature3(features);

        scrf::scrf_t graph;

        lattice::fst lat = lattice::load_lattice(lat_list);

        if (!lat_list) {
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

        scrf::composite_weight cost_aug_weight;
        cost_aug_weight.weights.push_back(std::make_shared<scrf::linear_weight>(
            scrf::linear_weight { param, graph_feat_func }));
        cost_aug_weight.weights.push_back(std::make_shared<scrf::cost::cost>(
            scrf::cost::cost(gold_path)));
        graph.weight_func = std::make_shared<scrf::composite_weight>(cost_aug_weight);
        graph.feature_func = std::make_shared<scrf::composite_feature>(graph_feat_func);

        std::shared_ptr<scrf::loss_func> loss_func;
        loss_func = std::make_shared<scrf::hinge_loss>(scrf::hinge_loss { gold_path, graph });
        real ell = loss_func->loss();

        std::cout << "loss: " << ell << std::endl;

        if (ell < -1e6) {
            std::cerr << "weird loss value. exit." << std::endl;
            exit(1);
        }

        if (ell < 0) {
            std::cout << "loss is less than zero.  skipping." << std::endl;
        }

        std::cout << std::endl;

        if (ell > 0) {
            auto param_grad = loss_func->param_grad();
            scrf::adagrad_update(param, param_grad, opt_data, step_size);

            if (i % save_every == 0) {
                scrf::save_param("param-last", param);
                scrf::save_param("opt-data-last", opt_data);
            }
        }

#if DEBUG_TOP_10
        if (i == 10) {
            break;
        }
#endif

        ++i;
    }

    scrf::save_param(output_param, param);
    scrf::save_param(output_opt_data, opt_data);

}

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "learn-lat",
        "Learn segmental CRF",
        {
            {"lat-list", "", true},
            {"gold-list", "", true},
            {"lm", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"step-size", "", true},
            {"features", "", true},
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
