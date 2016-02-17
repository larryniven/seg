#include "scrf/util.h"
#include "scrf/scrf.h"
#include "scrf/lm.h"
#include "scrf/lattice.h"
#include "speech/speech.h"
#include "scrf/loss.h"
#include "scrf/scrf_cost.h"
#include "scrf/scrf_feat.h"
#include "scrf/scrf_weight.h"
#include "scrf/scrf_util.h"
#include "scrf/make_feat.h"
#include <fstream>

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream lattice_batch;
    std::ifstream ground_truth_batch;
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

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "learn-lat",
        "Learn segmental CRF",
        {
            {"frame-batch", "", false},
            {"lattice-batch", "", true},
            {"ground-truth-batch", "", true},
            {"min-cost-path", "", false},
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

learning_env::learning_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    if (ebt::in(std::string("frame-batch"), args)) {
        frame_batch.open(args.at("frame-batch"));
    }

    lattice_batch.open(args.at("lattice-batch"));
    ground_truth_batch.open(args.at("ground-truth-batch"));

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

        std::vector<std::vector<real>> frames;

        if (frame_batch) {
            frames = speech::load_frame_batch(frame_batch);
        }

        lattice::fst ground_truth_lat = lattice::load_lattice(ground_truth_batch);

        if (!ground_truth_batch) {
            break;
        }

        lattice::fst lat = lattice::load_lattice(lattice_batch);

        if (!lattice_batch) {
            break;
        }

        std::cout << "ground truth: ";
        for (auto& e: ground_truth_lat.edges()) {
            std::cout << ground_truth_lat.output(e) << " ";
        }
        std::cout << std::endl;

        scrf::scrf_t ground_truth = scrf::make_gold_scrf(ground_truth_lat, lm);
        fst::path<scrf::scrf_t> ground_truth_path = scrf::make_ground_truth_path(ground_truth);

        scrf::scrf_t min_cost = scrf::make_lat_scrf(lat, lm);

        scrf::scrf_t gold;
        fst::path<scrf::scrf_t> gold_path;

        if (ebt::in(std::string("min-cost-path"), args)) {
            gold = min_cost;
            gold_path = scrf::make_min_cost_path(min_cost, ground_truth_path);
        } else {
            gold = ground_truth;
            gold_path = ground_truth_path;
        }
        gold_path.data->base_fst = &gold;

        scrf::composite_feature gold_feat_func = scrf::make_feat(features, frames, {});

        gold.weight_func = std::make_shared<scrf::composite_weight>(
            scrf::make_weight(features, param, gold_feat_func));
        gold.feature_func = std::make_shared<scrf::composite_feature>(gold_feat_func);

        scrf::composite_feature graph_feat_func = scrf::make_feat(features, frames, {});

        scrf::scrf_t graph = scrf::make_lat_scrf(lat, lm);

        graph.weight_func =
            graph.weight_func = std::make_shared<scrf::composite_weight>(
                scrf::make_weight(features, param, graph_feat_func))
            + std::make_shared<scrf::seg_cost>(
                scrf::make_overlap_cost(gold_path));
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
                scrf::save_param(param, "param-last");
                scrf::save_param(opt_data, "opt-data-last");
            }
        }

#if DEBUG_TOP_10
        if (i == 10) {
            break;
        }
#endif

        ++i;
    }

    scrf::save_param(param, output_param);
    scrf::save_param(opt_data, output_opt_data);

}

