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

#include "nn/nn-gpu.h"
#include "autodiff/autodiff-gpu.h"
#include "scrf/e2e-util.h"

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream ground_truth_batch;
    std::shared_ptr<lm::fst> lm;
    int min_seg;
    int max_seg;

    scrf::param_t param;
    scrf::param_t opt_data;

    double step_size;
    double nn_step_size;

    int save_every;

    nn::gpu::param_t nn_param;
    nn::gpu::opt_t nn_opt_data;
    nn::nn_t nn;

    std::string output_param;
    std::string output_opt_data;

    std::string output_nn_param;
    std::string output_nn_opt_data;

    std::vector<std::string> features;

    std::unordered_map<std::string, int> phone_id;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

nn::gpu::param_t nn_backprop(std::unordered_map<int, la::vector<double>> const& grad,
    std::vector<std::vector<double>> const& frames, nn::nn_t& nn, nn::gpu::param_t const& param);

struct hinge_loss {

    fst::path<scrf::scrf_t> const& gold_path;
    fst::path<scrf::scrf_t> cost_aug_path;
    std::unordered_map<std::string, int> const& phone_id;

    hinge_loss(fst::path<scrf::scrf_t> const& gold_path,
        scrf::scrf_t const& graph,
        std::unordered_map<std::string, int> const& phone_id);

    double loss() const;

    std::unordered_map<int, la::vector<double>> feat_grad() const;

    scrf::param_t param_grad() const;

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "learn-e2e",
        "Learn segmental CRF and neural network end to end",
        {
            {"frame-batch", "", true},
            {"ground-truth-batch", "", true},
            {"lm", "", true},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"min-cost-path", "Use min cost path for training", false},
            {"param", "", true},
            {"opt-data", "", true},
            {"step-size", "", true},
            {"nn-step-size", "", true},
            {"save-every", "", false},
            {"nn-param", "", true},
            {"nn-opt-data", "", true},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"output-nn-param", "", false},
            {"output-nn-opt-data", "", false},
            {"phone-id", "", true},
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
    step_size = std::stod(args.at("step-size"));
    nn_step_size = std::stod(args.at("nn-step-size"));

    if (ebt::in(std::string("save-every"), args)) {
        save_every = std::stoi(args.at("save-every"));
    } else {
        save_every = std::numeric_limits<int>::max();
    }

    nn_param = nn::load_param(args.at("nn-param"));
    nn_opt_data = nn::load_opt(args.at("nn-opt-data"));

    output_param = "param-last";
    if (ebt::in(std::string("output-param"), args)) {
        output_param = args.at("output-param");
    }

    output_opt_data = "opt-data-last";
    if (ebt::in(std::string("output-opt-data"), args)) {
        output_opt_data = args.at("output-opt-data");
    }

    output_nn_param = "nn-param-last";
    if (ebt::in(std::string("output-nn-param"), args)) {
        output_nn_param = args.at("output-nn-param");
    }

    output_nn_opt_data = "nn-opt-data-last";
    if (ebt::in(std::string("output-nn-opt-data"), args)) {
        output_nn_opt_data = args.at("output-nn-opt-data");
    }

    phone_id = scrf::load_phone_id(args.at("phone-id"));

    features = ebt::split(args.at("features"), ",");
}

void learning_env::run()
{
    std::shared_ptr<lm::fst> lm_output = scrf::erase_input(lm);

    int i = 1;

    while (1) {
        nn = nn::gpu::make_nn(nn_param);

        std::vector<std::vector<real>> frames;

        frames = speech::load_frame_batch(frame_batch);

        lattice::fst ground_truth_lat = lattice::load_lattice(ground_truth_batch);

        if (!frame_batch || !ground_truth_batch) {
            break;
        }

        std::vector<std::vector<real>> inputs = scrf::nn_feedforward(frames, nn);

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

        scrf::composite_feature gold_feat_func = scrf::make_feat(features, inputs, phone_id);

        gold.weight_func = std::make_shared<scrf::score::linear_score>(
            scrf::score::linear_score(param, std::make_shared<scrf::composite_feature>(gold_feat_func)));
        gold.feature_func = std::make_shared<scrf::composite_feature>(gold_feat_func);

        scrf::composite_feature graph_feat_func = scrf::make_feat(features, inputs, phone_id);

        scrf::scrf_t graph = scrf::make_graph_scrf(frames.size(), lm_output, min_seg, max_seg);

        scrf::seg_cost cost = scrf::make_overlap_cost(gold_path);
        scrf::score::linear_score score { param, std::make_shared<scrf::composite_feature>(graph_feat_func) };

        graph.weight_func = std::make_shared<scrf::score::linear_score>(score)
            + std::make_shared<scrf::seg_cost>(cost);
        graph.feature_func = std::make_shared<scrf::composite_feature>(graph_feat_func);

        hinge_loss loss_func { gold_path, graph, phone_id };
        real ell = loss_func.loss();

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

#if 0
        {
            nn::param_t nn_param2 = nn_param;
            nn_param2.label_bias(phone_id.at("<s>")) += 1e-8;
            nn::nn_t nn2 = nn::make_nn(nn_param2);

            std::vector<std::vector<real>> inputs2 = scrf::nn_feedforward(frames, nn2);

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

            scrf::composite_feature gold_feat_func { inputs2, phone_id };

            gold.weight_func = std::make_shared<scrf::score::linear_score>(
                scrf::score::linear_score(param, std::make_shared<scrf::composite_feature>(gold_feat_func)));
            gold.feature_func = std::make_shared<scrf::composite_feature>(gold_feat_func);

            scrf::composite_feature graph_feat_func { inputs2, phone_id };

            scrf::scrf_t graph = scrf::make_graph_scrf(frames.size(), lm_output, min_seg, max_seg);

            scrf::seg_cost cost = scrf::make_overlap_cost(gold_path);
            scrf::score::linear_score score { param, std::make_shared<scrf::composite_feature>(graph_feat_func) };

            graph.weight_func = std::make_shared<scrf::score::linear_score>(score)
                + std::make_shared<scrf::seg_cost>(cost);
            graph.feature_func = std::make_shared<scrf::composite_feature>(graph_feat_func);

            hinge_loss loss_func2 { gold_path, graph, phone_id };

            {
                auto edges1 = loss_func.gold_path.edges();
                auto edges2 = loss_func2.gold_path.edges();

                for (int i = 0; i < edges1.size(); ++i) {
                    int tail = std::get<0>(loss_func.gold_path.tail(edges1[i]));
                    int head = std::get<0>(loss_func.gold_path.head(edges1[i]));

                    auto& lat = *loss_func.gold_path.data->base_fst->fst->fst1;

                    std::cout << edges1[i] << " " << edges2[i] << " "
                        << lat.data->vertices.at(tail).time << " "
                        << lat.data->vertices.at(head).time << " "
                        << loss_func.gold_path.output(edges1[i]) << " "
                        << loss_func2.gold_path.weight(edges1[i]) << " "
                        << loss_func.gold_path.weight(edges2[i]) << " "
                        << loss_func2.gold_path.weight(edges1[i]) - loss_func.gold_path.weight(edges2[i])
                        << std::endl;
                }
            }

            {
                auto edges1 = loss_func.cost_aug_path.edges();
                auto edges2 = loss_func2.cost_aug_path.edges();

                for (int i = 0; i < edges1.size(); ++i) {
                    int tail = std::get<0>(loss_func.gold_path.tail(edges1[i]));
                    int head = std::get<0>(loss_func.gold_path.head(edges1[i]));

                    auto& lat = *loss_func.gold_path.data->base_fst->fst->fst1;

                    std::cout << edges1[i] << " " << edges2[i] << " "
                        << lat.data->vertices.at(tail).time << " "
                        << lat.data->vertices.at(head).time << " "
                        << loss_func2.gold_path.weight(edges1[i]) << " "
                        << loss_func.gold_path.weight(edges2[i]) << " "
                        << loss_func2.gold_path.weight(edges1[i]) - loss_func.gold_path.weight(edges2[i])
                        << std::endl;
                }
            }

            double ell2 = loss_func2.loss();

            std::cout << "numeric grad: " << (ell2 - ell) / 1e-8
                << " " << ell2 << " " << ell << " " << ell2 - ell << std::endl;
        }
#endif

        if (ell > 0) {
            std::unordered_map<int, la::vector<double>> feat_grad = loss_func.feat_grad();
            nn::gpu::param_t nn_grad = nn_backprop(feat_grad, frames, nn, nn_param);

            // std::cout << "calc grad: " << nn_grad.label_bias(phone_id.at("<s>")) << std::endl;

            nn::gpu::adagrad_update(nn_param, nn_grad, nn_opt_data, nn_step_size);

            scrf::param_t grad = loss_func.param_grad();
            scrf::adagrad_update(param, grad, opt_data, step_size);

            if (i % save_every == 0) {
                scrf::save_param(param, "param-last");
                scrf::save_param(opt_data, "opt-data-last");

                nn::save_param(nn::gpu::to_host(nn_param), "nn-param-last");
                nn::save_opt(nn::gpu::to_host(nn_opt_data), "nn-opt-data-last");
            }
        }

        std::cout << std::endl;

#if DEBUG_TOP_10
        if (i == 10) {
            break;
        }
#endif

        ++i;
    }

    scrf::save_param(param, output_param);
    scrf::save_param(opt_data, output_opt_data);

    nn::save_param(nn::gpu::to_host(nn_param), output_nn_param);
    nn::save_opt(nn::gpu::to_host(nn_opt_data), output_nn_opt_data);

}

hinge_loss::hinge_loss(
    fst::path<scrf::scrf_t> const& gold_path,
    scrf::scrf_t const& graph,
    std::unordered_map<std::string, int> const& phone_id)
    : gold_path(gold_path), phone_id(phone_id)
{
    auto order = scrf::topo_order(graph);
    cost_aug_path = scrf::shortest_path(graph, order);
}

double hinge_loss::loss() const
{
    double gold_score = 0;
    for (auto& e: gold_path.edges()) {
        gold_score += gold_path.weight(e);
    }

    std::cout << "gold score: " << gold_score << std::endl;

    scrf::seg_cost cost = scrf::make_overlap_cost(gold_path);

    double cost_aug_weight = 0;
    double cost_aug_cost = 0;
    std::cout << "cost aug path: ";
    for (auto& e: cost_aug_path.edges()) {
        cost_aug_weight += cost_aug_path.weight(e);
        cost_aug_cost += cost(*cost_aug_path.data->base_fst->fst, e);
        std::cout << cost_aug_path.output(e) << " ";
    }
    std::cout << std::endl;
    std::cout << "cost aug cost: " << cost_aug_cost << std::endl;
    std::cout << "cost aug score: " << cost_aug_weight - cost_aug_cost << std::endl;

    return cost_aug_weight - gold_score;
}

std::unordered_map<int, la::vector<double>> hinge_loss::feat_grad() const
{
    std::unordered_map<int, la::vector<double>> result;

    lattice::fst& gold_lat = *gold_path.data->base_fst->fst->fst1;
    for (auto& e: gold_path.edges()) {
        int tail = std::get<0>(gold_path.tail(e));
        int tail_time = gold_lat.data->vertices.at(tail).time;

        // result[tail_time].resize(phone_id.size());
        // result[tail_time](phone_id.at(gold_path.output(e))) -= 1;

        int head = std::get<0>(gold_path.head(e));
        int head_time = gold_lat.data->vertices.at(head).time;

        // result[head_time - 1].resize(phone_id.size());
        // result[head_time - 1](phone_id.at(gold_path.output(e))) -= 1;

        // result[int((tail_time + head_time - 1) / 2)].resize(phone_id.size());
        // result[int((tail_time + head_time - 1) / 2)](phone_id.at(gold_path.output(e))) -= 1;

        for (int i = tail_time; i < head_time; ++i) {
            result[i].resize(phone_id.size());
            result[i](phone_id.at(gold_path.output(e))) -= 1;
        }
    }

    lattice::fst& cost_aug_lat = *cost_aug_path.data->base_fst->fst->fst1;
    for (auto& e: cost_aug_path.edges()) {
        int tail = std::get<0>(cost_aug_path.tail(e));
        int tail_time = cost_aug_lat.data->vertices.at(tail).time;

        // result[tail_time].resize(phone_id.size());
        // result[tail_time](phone_id.at(cost_aug_path.output(e))) += 1;

        int head = std::get<0>(cost_aug_path.head(e));
        int head_time = cost_aug_lat.data->vertices.at(head).time;

        // result[head_time - 1].resize(phone_id.size());
        // result[head_time - 1](phone_id.at(cost_aug_path.output(e))) += 1;

        // result[int((tail_time + head_time - 1) / 2)].resize(phone_id.size());
        // result[int((tail_time + head_time - 1) / 2)](phone_id.at(cost_aug_path.output(e))) += 1;

        for (int i = tail_time; i < head_time; ++i) {
            result[i].resize(phone_id.size());
            result[i](phone_id.at(cost_aug_path.output(e))) += 1;
        }
    }

    return result;
}

scrf::param_t hinge_loss::param_grad() const
{
    scrf::param_t result;

    scrf::scrf_t const& gold_scrf = *gold_path.data->base_fst;
    for (auto& e: gold_path.edges()) {
        scrf::feat_t f;
        (*gold_scrf.feature_func)(f, *gold_scrf.fst, e);
        scrf::isub(result, to_param(std::move(f)));
    }

    scrf::scrf_t const& graph_scrf = *cost_aug_path.data->base_fst;
    for (auto& e: cost_aug_path.edges()) {
        scrf::feat_t f;
        (*graph_scrf.feature_func)(f, *graph_scrf.fst, e);
        scrf::iadd(result, to_param(std::move(f)));
    }

    return result;
}

nn::gpu::param_t nn_backprop(std::unordered_map<int, la::vector<double>> const& grad,
    std::vector<std::vector<double>> const& frames, nn::nn_t& nn, nn::gpu::param_t const& param)
{
    int dim = frames.front().size();
    nn::gpu::param_t result;
    nn::gpu::resize_as(result, param);

    la::vector<double> input_block;
    input_block.resize(frames.size() * 11 * dim);

    for (auto& p: grad) {
        int i = p.first;

        std::vector<double> input;

        for (int j = i - 5; j <= i + 5; ++j) {
            if (j < 0 || j >= frames.size()) {
                input.resize(input.size() + dim);
            } else {
                input.insert(input.end(), frames[j].begin(), frames[j].end());
            }
        }

        std::copy(input.begin(), input.end(), input_block.data() + 11 * dim * i);
    }

    la::gpu::vector<double> input_gpu_block { input_block };

    for (auto& p: grad) {
        int i = p.first;

        nn.hidden[0]->output = std::make_shared<la::gpu::vector_view<double>>(
            la::gpu::vector_view<double>(input_gpu_block.data() + 11 * dim * i, 11 * dim));

        autodiff::eval(nn.output, autodiff::gpu::eval_funcs);

        nn.output->grad = std::make_shared<la::gpu::vector<double>>(
            la::gpu::vector<double>(p.second));

        autodiff::grad(nn.output, autodiff::gpu::grad_funcs);

        nn::gpu::iadd(result, nn::gpu::copy_grad(nn));

        nn::gpu::zero_grad(nn);
    }

    return result;
}

