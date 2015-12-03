#include "scrf/util.h"
#include "scrf/scrf.h"
#include "scrf/lm.h"
#include "scrf/lattice.h"
#include "speech/speech.h"
#include "scrf/nn.h"
#include "scrf/loss.h"
#include "scrf/scrf_cost.h"
#include "scrf/scrf_feat.h"
#include "scrf/scrf_weight.h"
#include "scrf/scrf_util.h"
#include <fstream>

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream ground_truth_batch;
    std::shared_ptr<lm::fst> lm;
    int min_seg;
    int max_seg;
    scrf::param_t param;
    scrf::param_t opt_data;
    real step_size;

    int save_every;

    std::string output_param;
    std::string output_opt_data;

    double norm;

    std::vector<std::string> features;

    int beam_width;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "learn",
        "Learn segmental CRF",
        {
            {"frame-batch", "", false},
            {"ground-truth-batch", "", true},
            {"lm", "", true},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"min-cost-path", "Use min cost path for training", false},
            {"param", "", true},
            {"opt-data", "", true},
            {"step-size", "", true},
            {"features", "", true},
            {"save-every", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"loss", "", true},
            {"beam-width", "", false},
            {"norm", "", false}
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

    ground_truth_batch.open(args.at("ground-truth-batch"));

    lm = std::make_shared<lm::fst>(lm::load_arpa_lm(args.at("lm")));

    min_seg = 1;
    if (ebt::in(std::string("min-seg"), args)) {
        min_seg = std::stoi(args.at("min-seg"));
    }

    max_seg = 20;
    if (ebt::in(std::string("max-seg"), args)) {
        max_seg = std::stoi(args.at("max-seg"));
    }

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

    if (ebt::in(std::string("beam-width"), args)) {
        beam_width = std::stoi(args.at("beam-width"));
    }

    if (ebt::in(std::string("norm"), args)) {
        norm = std::stod(args.at("norm"));
    }
}

void learning_env::run()
{
    std::shared_ptr<lm::fst> lm_output = scrf::erase_input(lm);

    int i = 1;

    while (1) {

        std::vector<std::vector<real>> frames;

        if (frame_batch) {
            frames = speech::load_frames_batch(frame_batch);
        }

        lattice::fst ground_truth_lat = lattice::load_lattice(ground_truth_batch);

        if (!ground_truth_batch) {
            break;
        }

        std::cout << ground_truth_lat.data->name << std::endl;

        std::cout << "ground truth: ";
        for (auto& e: ground_truth_lat.edges()) {
            std::cout << ground_truth_lat.output(e) << " ";
        }
        std::cout << std::endl;

        scrf::scrf_t ground_truth = scrf::make_gold_scrf(ground_truth_lat, lm);
        fst::path<scrf::scrf_t> ground_truth_path = scrf::make_ground_truth_path(ground_truth);

        scrf::scrf_t min_cost = scrf::make_graph_scrf(frames.size(), lm_output, min_seg, max_seg);

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

        scrf::composite_feature gold_feat_func = scrf::make_feature2(features, frames);

        gold.weight_func = std::make_shared<scrf::composite_weight>(
            scrf::make_weight(param, gold_feat_func));
        gold.feature_func = std::make_shared<scrf::composite_feature>(gold_feat_func);

        scrf::composite_feature graph_feat_func = scrf::make_feature2(features, frames);

        scrf::scrf_t graph = scrf::make_graph_scrf(frames.size(), lm_output, min_seg, max_seg);

        graph.weight_func =
            std::make_shared<scrf::composite_weight>(
                scrf::make_weight(param, graph_feat_func))
            + std::make_shared<scrf::seg_cost>(
                scrf::make_overlap_cost(gold_path));
        graph.feature_func = std::make_shared<scrf::composite_feature>(graph_feat_func);

        std::shared_ptr<scrf::loss_func> loss_func;
        if (args.at("loss") == "hinge") {
            loss_func = std::make_shared<scrf::hinge_loss>(scrf::hinge_loss { gold_path, graph });
        } else if (args.at("loss") == "hinge-beam") {
            loss_func = std::make_shared<scrf::hinge_loss_beam>(scrf::hinge_loss_beam { gold_path, graph, beam_width });
        } else {
            std::cout << "unknown loss function " << args.at("loss") << std::endl;
            exit(1);
        }
        real ell = loss_func->loss();

        std::cout << "gold segs: " << gold_path.edges().size()
            << " frames: " << frames.size() << std::endl;
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

            if (ebt::in(std::string("norm"), args)) {
                double n = scrf::norm(param);

                if (n > norm) {
                    param *= norm / n;
                }
            }

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

