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

struct inference_args {
    int min_seg;
    int max_seg;
    scrf::first_order::param_t param;
    std::unordered_map<std::string, int> label_id;
    std::vector<std::string> id_label;
    std::vector<int> labels;
    std::vector<std::string> features;
    std::unordered_map<std::string, std::string> args;
};

struct sample {
    std::vector<std::vector<double>> frames;

    scrf::first_order::feat_dim_alloc graph_alloc;

    scrf::first_order::scrf_t graph;
    fst::path<scrf::first_order::scrf_t> graph_path;

    sample(inference_args const& args);
};

void make_graph(sample& s, inference_args const& i_args);

struct learning_args
    : public inference_args {

    scrf::first_order::param_t opt_data;
    double step_size;
    double momentum;
    std::vector<int> sils;
};

struct learning_sample
    : public sample {

    ilat::fst ground_truth_fst;

    scrf::first_order::feat_dim_alloc gold_alloc;

    scrf::first_order::scrf_t ground_truth;
    fst::path<scrf::first_order::scrf_t> ground_truth_path;

    scrf::first_order::scrf_t gold;
    fst::path<scrf::first_order::scrf_t> gold_path;

    std::shared_ptr<scrf::first_order::seg_cost> cost;

    learning_sample(learning_args const& args);
};

learning_args parse_learning_args(
    std::unordered_map<std::string, std::string> const& args);

void make_gold(learning_sample& s, learning_args const& l_args);
void make_min_cost_gold(learning_sample& s, learning_args const& l_args);

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream ground_truth_batch;

    int save_every;

    std::string output_param;
    std::string output_opt_data;

    learning_args l_args;

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

    ground_truth_batch.open(args.at("gold-batch"));

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

    l_args = parse_learning_args(args);
}

void learning_env::run()
{
    int i = 1;

    while (1) {

        learning_sample s { l_args };

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
            make_min_cost_gold(s, l_args);
        } else {
            make_gold(s, l_args);
        }

        s.cost = std::make_shared<scrf::first_order::seg_cost>(
            scrf::first_order::make_overlap_cost(s.ground_truth_path, l_args.sils));

        double gold_cost = 0;
    
        std::cout << "gold path: ";
        for (auto& e: s.gold_path.edges()) {
            std::cout << l_args.id_label[s.gold_path.output(e)] << " ";
            gold_cost += (*s.cost)(*s.gold.fst, e);
        }
        std::cout << std::endl;
    
        std::cout << "gold cost: " << gold_cost << std::endl;

        make_graph(s, l_args);

        std::shared_ptr<scrf::first_order::loss_func> loss_func;

        if (args.at("loss") == "hinge-loss") {
            scrf::first_order::composite_weight& graph_weight_func
                = *dynamic_cast<scrf::first_order::composite_weight*>(s.graph.weight_func.get());
            graph_weight_func.weights.push_back(s.cost);

            loss_func = std::make_shared<scrf::first_order::hinge_loss>(
                scrf::first_order::hinge_loss { s.gold_path, s.graph });

            scrf::first_order::hinge_loss const& loss = *dynamic_cast<scrf::first_order::hinge_loss*>(loss_func.get());

            real gold_weight = 0;

            std::cout << "gold: ";
            for (auto& e: s.gold_path.edges()) {
                std::cout << l_args.id_label[s.gold_path.output(e)] << " ";
                gold_weight += s.gold_path.weight(e);
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
        } else if (args.at("loss") == "log-loss") {
            loss_func = std::make_shared<scrf::first_order::log_loss>(
                scrf::first_order::log_loss { s.gold_path, s.graph });
        } else if (args.at("loss") == "filtering-loss") {
            if (!ebt::in(std::string("alpha"), args)) {
                std::cerr << "--alpha is required" << std::endl;
            }

            double alpha = std::stod(args.at("alpha"));

            loss_func = std::make_shared<scrf::first_order::filtering_loss>(
                scrf::first_order::filtering_loss { s.gold_path, s.graph, alpha });
        } else {
            std::cout << "unknown loss function " << args.at("loss") << std::endl;
            exit(1);
        }

        std::cout << "gold segs: " << s.gold_path.edges().size()
            << " frames: " << s.frames.size() << std::endl;

        double ell = loss_func->loss();

        std::cout << "loss: " << ell << std::endl;

        scrf::first_order::param_t param_grad;

        if (ell > 0) {
            param_grad = loss_func->param_grad();

            std::cout << "analytic grad: " << param_grad.class_vec[l_args.label_id.at("</s>")](0) << std::endl;
        }

        if (ell < 0) {
            std::cout << "loss is less than zero.  skipping." << std::endl;
        }

        std::cout << std::endl;

        if (ebt::in(std::string("momentum"), args)) {
            scrf::first_order::const_step_update_momentum(l_args.param, param_grad,
               l_args.opt_data, l_args.momentum, l_args.step_size);
        } else {
            scrf::first_order::adagrad_update(l_args.param, param_grad, l_args.opt_data, l_args.step_size);
        }

        if (i % save_every == 0) {
            scrf::first_order::save_param(l_args.param, "param-last");
            scrf::first_order::save_param(l_args.opt_data, "opt-data-last");
        }

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        ++i;
    }

    scrf::first_order::save_param(l_args.param, output_param);
    scrf::first_order::save_param(l_args.opt_data, output_opt_data);

}

sample::sample(inference_args const& args)
    : graph_alloc(args.labels)
{
}

void make_graph(sample& s, inference_args const& i_args)
{
    s.graph = scrf::first_order::make_graph_scrf(s.frames.size(),
        i_args.labels, i_args.min_seg, i_args.max_seg);

    scrf::first_order::composite_feature graph_feat_func
        = scrf::first_order::make_feat(s.graph_alloc, i_args.features, s.frames, i_args.args);

    scrf::first_order::composite_weight weight;
    weight.weights.push_back(std::make_shared<scrf::first_order::score::cached_linear_score>(
        scrf::first_order::score::cached_linear_score(i_args.param,
            std::make_shared<scrf::first_order::composite_feature>(graph_feat_func),
            *s.graph.fst)));

    s.graph.weight_func = std::make_shared<scrf::first_order::composite_weight>(weight);
    s.graph.feature_func = std::make_shared<scrf::first_order::composite_feature>(graph_feat_func);
}

learning_args parse_learning_args(
    std::unordered_map<std::string, std::string> const& args)
{
    learning_args l_args;

    l_args.args = args;

    l_args.min_seg = 1;
    if (ebt::in(std::string("min-seg"), args)) {
        l_args.min_seg = std::stoi(args.at("min-seg"));
    }

    l_args.max_seg = 20;
    if (ebt::in(std::string("max-seg"), args)) {
        l_args.max_seg = std::stoi(args.at("max-seg"));
    }

    l_args.param = scrf::first_order::load_param(args.at("param"));
    l_args.opt_data = scrf::first_order::load_param(args.at("opt-data"));
    l_args.step_size = std::stod(args.at("step-size"));

    l_args.momentum = -1;
    if (ebt::in(std::string("momentum"), args)) {
        l_args.momentum = std::stod(args.at("momentum"));
        assert(0 <= l_args.momentum && l_args.momentum <= 1);
    }

    l_args.features = ebt::split(args.at("features"), ",");

    l_args.label_id = scrf::load_phone_id(args.at("label"));

    l_args.id_label.resize(l_args.label_id.size());
    for (auto& p: l_args.label_id) {
        l_args.labels.push_back(p.second);
        l_args.id_label[p.second] = p.first;
    }

    l_args.sils.push_back(l_args.label_id.at("<s>"));
    l_args.sils.push_back(l_args.label_id.at("</s>"));
    l_args.sils.push_back(l_args.label_id.at("sil"));

    return l_args;
}

learning_sample::learning_sample(learning_args const& args)
    : sample(args), gold_alloc(args.labels)
{
}

void make_gold(learning_sample& s, learning_args const& l_args)
{
    s.ground_truth.fst = std::make_shared<ilat::fst>(s.ground_truth_fst);
    s.ground_truth_path = scrf::first_order::make_ground_truth_path(s.ground_truth);

    s.gold = s.ground_truth;
    s.gold_path = s.ground_truth_path;
    s.gold_path.data->base_fst = &s.gold;

    scrf::first_order::composite_feature gold_feat_func
        = scrf::first_order::make_feat(s.gold_alloc, l_args.features, s.frames, l_args.args);

    s.gold.weight_func = std::make_shared<scrf::first_order::score::cached_linear_score>(
        scrf::first_order::score::cached_linear_score(l_args.param,
        std::make_shared<scrf::first_order::composite_feature>(gold_feat_func),
        *s.gold.fst));
    s.gold.feature_func = std::make_shared<scrf::first_order::composite_feature>(gold_feat_func);
}

void make_min_cost_gold(learning_sample& s, learning_args const& l_args)
{
    s.ground_truth.fst = std::make_shared<ilat::fst>(s.ground_truth_fst);
    s.ground_truth_path = scrf::first_order::make_ground_truth_path(s.ground_truth);

    s.gold = scrf::first_order::make_graph_scrf(s.frames.size(), l_args.labels, l_args.min_seg, l_args.max_seg);
    s.gold_path = scrf::first_order::make_min_cost_path(s.gold, s.ground_truth_path, l_args.sils);
    s.gold_path.data->base_fst = &s.gold;

    scrf::first_order::composite_feature gold_feat_func
        = scrf::first_order::make_feat(s.gold_alloc, l_args.features, s.frames, l_args.args);

    s.gold.weight_func = std::make_shared<scrf::first_order::score::cached_linear_score>(
        scrf::first_order::score::cached_linear_score(l_args.param,
        std::make_shared<scrf::first_order::composite_feature>(gold_feat_func),
        *s.gold.fst));
    s.gold.feature_func = std::make_shared<scrf::first_order::composite_feature>(gold_feat_func);
}

