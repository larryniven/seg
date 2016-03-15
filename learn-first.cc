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
    std::ifstream ground_truth_batch;
    int min_seg;
    int max_seg;
    scrf::first_order::param_t param;
    scrf::first_order::param_t opt_data;
    double step_size;
    double momentum;

    int save_every;

    std::string output_param;
    std::string output_opt_data;

    std::unordered_map<std::string, int> label_id;
    std::vector<std::string> id_label;
    std::vector<int> labels;
    std::vector<int> sils;

    std::vector<std::string> features;

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
            {"frame-batch", "", true},
            {"gold-batch", "", true},
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

    ground_truth_batch.open(args.at("gold-batch"));

    min_seg = 1;
    if (ebt::in(std::string("min-seg"), args)) {
        min_seg = std::stoi(args.at("min-seg"));
    }

    max_seg = 20;
    if (ebt::in(std::string("max-seg"), args)) {
        max_seg = std::stoi(args.at("max-seg"));
    }

    param = scrf::first_order::load_param(args.at("param"));
    opt_data = scrf::first_order::load_param(args.at("opt-data"));
    step_size = std::stod(args.at("step-size"));

    if (ebt::in(std::string("momentum"), args)) {
        momentum = std::stod(args.at("momentum"));
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

    label_id = scrf::load_phone_id(args.at("label"));

    id_label.resize(label_id.size());
    for (auto& p: label_id) {
        labels.push_back(p.second);
        id_label[p.second] = p.first;
    }

    sils.push_back(label_id.at("<s>"));
    sils.push_back(label_id.at("</s>"));
    sils.push_back(label_id.at("sil"));
}

void learning_env::run()
{
    int i = 1;

    while (1) {

        std::vector<std::vector<double>> frames;

        frames = speech::load_frame_batch(frame_batch);

        ilat::fst ground_truth_lat = ilat::load_lattice(ground_truth_batch, label_id);

        if (!ground_truth_batch) {
            break;
        }

        std::cout << ground_truth_lat.data->name << std::endl;

        std::cout << "ground truth: ";
        for (auto& e: ground_truth_lat.edges()) {
            std::cout << id_label[ground_truth_lat.output(e)] << " ";
        }
        std::cout << std::endl;

        double ell;
        scrf::first_order::param_t param_grad;

        scrf::first_order::scrf_t ground_truth;
        ground_truth.fst = std::make_shared<ilat::fst>(ground_truth_lat);
        fst::path<scrf::first_order::scrf_t> ground_truth_path
            = scrf::first_order::make_ground_truth_path(ground_truth);

        scrf::first_order::scrf_t min_cost = scrf::first_order::make_graph_scrf(frames.size(),
            labels, min_seg, max_seg);

        scrf::first_order::scrf_t gold;
        fst::path<scrf::first_order::scrf_t> gold_path;

        if (ebt::in(std::string("min-cost-path"), args)) {
            gold = min_cost;
            gold_path = scrf::first_order::make_min_cost_path(min_cost, ground_truth_path, sils);

            double min_cost_path_weight = 0;
    
            std::cout << "min cost path: ";
            for (auto& e: gold_path.edges()) {
                std::cout << id_label[gold_path.output(e)] << " ";
                min_cost_path_weight += gold_path.weight(e);
            }
            std::cout << std::endl;
    
            std::cout << "cost: " << min_cost_path_weight << std::endl;
    
        } else {
            gold = ground_truth;
            gold_path = ground_truth_path;
        }
        gold_path.data->base_fst = &gold;

        scrf::first_order::feat_dim_alloc gold_alloc { labels };

        scrf::first_order::composite_feature gold_feat_func
            = scrf::first_order::make_feat(gold_alloc, features, frames);

        gold.weight_func = std::make_shared<scrf::first_order::score::linear_score>(
            scrf::first_order::score::linear_score(param,
            std::make_shared<scrf::first_order::composite_feature>(gold_feat_func)));
        gold.feature_func = std::make_shared<scrf::first_order::composite_feature>(gold_feat_func);

        scrf::first_order::feat_dim_alloc graph_alloc { labels };

        scrf::first_order::composite_feature graph_feat_func
            = scrf::first_order::make_feat(graph_alloc, features, frames);

        scrf::first_order::scrf_t graph = scrf::first_order::make_graph_scrf(frames.size(),
            labels, min_seg, max_seg);

        scrf::first_order::composite_weight weight;
        weight.weights.push_back(std::make_shared<scrf::first_order::score::linear_score>(
            scrf::first_order::score::linear_score(param,
                std::make_shared<scrf::first_order::composite_feature>(graph_feat_func))));
        weight.weights.push_back(std::make_shared<scrf::first_order::seg_cost>(
            scrf::first_order::make_overlap_cost(gold_path, sils)));

        graph.weight_func = std::make_shared<scrf::first_order::composite_weight>(weight);
        graph.feature_func = std::make_shared<scrf::first_order::composite_feature>(graph_feat_func);

        std::shared_ptr<scrf::first_order::loss_func> loss_func;

        if (args.at("loss") == "hinge") {
            loss_func = std::make_shared<scrf::first_order::hinge_loss>(
                scrf::first_order::hinge_loss { gold_path, graph });

            scrf::first_order::hinge_loss const& loss = *dynamic_cast<scrf::first_order::hinge_loss*>(loss_func.get());

            real gold_score = 0;

            std::cout << "gold: ";
            for (auto& e: gold_path.edges()) {
                std::cout << id_label[gold_path.output(e)] << " ";
                gold_score += gold_path.weight(e);
            }
            std::cout << std::endl;

            std::cout << "gold score: " << gold_score << std::endl;

            real graph_score = 0;

            std::cout << "cost aug: ";
            for (auto& e: loss.graph_path.edges()) {
                std::cout << id_label[graph.output(e)] << " ";
                graph_score += loss.graph_path.weight(e);
            }
            std::cout << std::endl;

            std::cout << "cost aug score: " << graph_score << std::endl; 

        } else {
            std::cout << "unknown loss function " << args.at("loss") << std::endl;
            exit(1);
        }

        std::cout << "gold segs: " << gold_path.edges().size()
            << " frames: " << frames.size() << std::endl;

        ell = loss_func->loss();

        if (ell > 0) {
            param_grad = loss_func->param_grad();
        }

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
            if (ebt::in(std::string("momentum"), args)) {
                scrf::first_order::const_step_update_momentum(param, std::move(param_grad),
                   opt_data, momentum, step_size);
            } else {
                scrf::first_order::adagrad_update(param, param_grad, opt_data, step_size);
            }

            if (i % save_every == 0) {
                scrf::first_order::save_param(param, "param-last");
                scrf::first_order::save_param(opt_data, "opt-data-last");
            }
        }

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        ++i;
    }

    scrf::first_order::save_param(param, output_param);
    scrf::first_order::save_param(opt_data, output_opt_data);

}

