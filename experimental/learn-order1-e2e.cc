#include "scrf/experimental/iscrf_e2e.h"
#include "scrf/experimental/loss.h"
#include "scrf/experimental/scrf_weight.h"
#include "autodiff/autodiff.h"
#include "nn/lstm.h"
#include <fstream>
#include <random>

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream gold_batch;

    std::ifstream lattice_batch;

    int save_every;

    std::string output_param;
    std::string output_opt_data;

    std::string output_nn_param;
    std::string output_nn_opt_data;

    scrf::e2e::learning_args l_args;

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
            {"lattice-batch", "", false},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"param", "", true},
            {"opt-data", "", true},
            {"step-size", "", true},
            {"nn-param", "", true},
            {"nn-opt-data", "", true},
            {"momentum", "", false},
            {"decay", "", false},
            {"features", "", true},
            {"save-every", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"output-nn-param", "", false},
            {"output-nn-opt-data", "", false},
            {"loss", "", true},
            {"cost-scale", "", false},
            {"label", "", true},
            {"rnndrop-prob", "", false},
            {"rnndrop-seed", "", false},
            {"subsample-freq", "", false},
            {"subsample-shift", "", false},
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

    gold_batch.open(args.at("gold-batch"));

    if (ebt::in(std::string("lattice-batch"), args)) {
        lattice_batch.open(args.at("lattice-batch"));
    }

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

    output_nn_param = "nn-param-last";
    if (ebt::in(std::string("output-nn-param"), args)) {
        output_nn_param = args.at("output-nn-param");
    }

    output_nn_opt_data = "nn-opt-data-last";
    if (ebt::in(std::string("output-nn-opt-data"), args)) {
        output_nn_opt_data = args.at("output-nn-opt-data");
    }

    scrf::e2e::parse_learning_args(l_args, args);
}

void learning_env::run()
{
    int i = 1;

    std::default_random_engine gen { l_args.rnndrop_seed };

    while (1) {

        scrf::e2e::learning_sample s { l_args };

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        s.gold_segs = scrf::load_segments(gold_batch, l_args.label_id);

        if (!gold_batch) {
            break;
        }

        autodiff::computation_graph comp_graph;

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        for (auto& f: frames) {
            frame_ops.push_back(comp_graph.var(la::vector<double>(f)));
        }

        std::vector<std::shared_ptr<autodiff::op_t>> subsampled_input;
        if (l_args.subsample_freq > 1) {
            subsampled_input = rnn::subsample_input(frame_ops,
                l_args.subsample_freq, l_args.subsample_shift);
        } else {
            subsampled_input = frame_ops;
        }

        lstm::dblstm_feat_nn_t nn = lstm::make_dblstm_feat_nn(
            comp_graph, l_args.nn_param, subsampled_input);

        if (ebt::in(std::string("rnndrop-prob"), args)) {
            lstm::apply_random_mask(nn, l_args.nn_param, gen, l_args.rnndrop_prob);
        }

        rnn::pred_nn_t pred_nn = rnn::make_pred_nn(comp_graph,
            l_args.pred_param, nn.layer.back().output);

        std::vector<std::shared_ptr<autodiff::op_t>> upsampled_output;
        if (l_args.subsample_freq > 1) {
             upsampled_output = rnn::upsample_output(pred_nn.logprob,
                 l_args.subsample_freq, l_args.subsample_shift, frames.size());
        } else {
             upsampled_output = pred_nn.logprob;
        }

        auto order = autodiff::topo_order(upsampled_output);
        autodiff::eval(order, autodiff::eval_funcs);

        std::vector<std::vector<double>> inputs;
        for (auto& o: upsampled_output) {
            auto& f = autodiff::get_output<la::vector<double>>(o);
            inputs.push_back(std::vector<double> {f.data(), f.data() + f.size()});
        }

        s.frames = inputs;

        if (ebt::in(std::string("lattice-batch"), args)) {
            ilat::fst lat = ilat::load_lattice(lattice_batch, l_args.label_id);

            if (!lattice_batch) {
                std::cerr << "error reading " << args.at("lattice-batch") << std::endl;
                exit(1);
            }

            scrf::e2e::make_lattice(lat, s, l_args);
        } else {
            scrf::e2e::make_graph(s, l_args);
        }

        scrf::e2e::make_min_cost_gold(s, l_args);

        s.cost = std::make_shared<scrf::mul<ilat::fst>>(scrf::mul<ilat::fst>(
            std::make_shared<scrf::seg_cost<ilat::fst>>(
                scrf::make_overlap_cost<ilat::fst>(s.gold_segs, l_args.sils)),
            l_args.cost_scale));

        double gold_cost = 0;
    
        std::cout << "gold path: ";
        for (auto& e: s.gold->edges()) {
            std::cout << l_args.id_label[s.gold->output(e)] << " ";
            gold_cost += (*s.cost)(*s.gold->fst, e);
        }
        std::cout << std::endl;
    
        std::cout << "gold cost: " << gold_cost << std::endl;

        scrf::e2e::parameterize(s, l_args);

        std::shared_ptr<scrf::loss_func_with_frame_grad<scrf::dense_vec>> loss_func;

        if (args.at("loss") == "hinge-loss") {
            scrf::composite_weight<ilat::fst>& graph_weight_func
                = *dynamic_cast<scrf::composite_weight<ilat::fst>*>(s.graph.weight_func.get());
            graph_weight_func.weights.push_back(s.cost);

            using hinge_loss = scrf::hinge_loss_with_frame_grad<
                scrf::e2e::iscrf, scrf::dense_vec,
                scrf::e2e::iscrf_path_maker>;

            loss_func = std::make_shared<hinge_loss>(
                hinge_loss { *s.gold, s.graph });

            hinge_loss const& loss
                = *dynamic_cast<hinge_loss*>(loss_func.get());

            double gold_weight = 0;

            std::cout << "gold: ";
            for (auto& e: s.gold->edges()) {
                std::cout << l_args.id_label[s.gold->output(e)] << " ";
                gold_weight += s.gold->weight(e);
            }
            std::cout << std::endl;

            std::cout << "gold score: " << gold_weight << std::endl;

            double graph_weight = 0;

            std::cout << "cost aug: ";
            for (auto& e: loss.graph_path->edges()) {
                std::cout << l_args.id_label[loss.graph.output(e)] << " ";
                graph_weight += loss.graph_path->weight(e);
            }
            std::cout << std::endl;

            std::cout << "cost aug score: " << graph_weight << std::endl;
        } else {
            std::cout << "unknown loss function " << args.at("loss") << std::endl;
            exit(1);
        }

        std::cout << "gold segs: " << s.gold->edges().size()
            << " frames: " << s.frames.size() << std::endl;

        double ell = loss_func->loss();

        std::cout << "loss: " << ell << std::endl;

        scrf::dense_vec param_grad;
        lstm::dblstm_feat_param_t nn_param_grad;
        rnn::pred_param_t pred_grad;

        if (ell > 0) {
            param_grad = loss_func->param_grad();

            std::vector<std::vector<double>> frame_grad;
            frame_grad.resize(inputs.size());
            for (int i = 0; i < inputs.size(); ++i) {
                frame_grad[i].resize(inputs[i].size());
            }

            loss_func->frame_grad(frame_grad, l_args.param);

            for (int i = 0; i < frame_grad.size(); ++i) {
                upsampled_output[i]->grad = std::make_shared<la::vector<double>>(
                    frame_grad[i]);
            }

            autodiff::grad(order, autodiff::grad_funcs);

            nn_param_grad = lstm::copy_dblstm_feat_grad(nn);
            pred_grad = rnn::copy_grad(pred_nn);

            std::cout << "analytical grad: "
                << pred_grad.softmax_weight(0, 0) << std::endl;
        }

#if 0
        {
            scrf::e2e::learning_args l_args2 = l_args;
            l_args2.pred_param.softmax_weight(0, 0) += 1e-8;

            scrf::e2e::learning_sample s2 { l_args2 };

            autodiff::computation_graph comp_graph2;

            std::vector<std::shared_ptr<autodiff::op_t>> frame_ops2;
            for (auto& f: frames) {
                frame_ops2.push_back(comp_graph2.var(la::vector<double>(f)));
            }

            lstm::dblstm_feat_nn_t nn2 = lstm::make_dblstm_feat_nn(
                comp_graph2, l_args2.nn_param, frame_ops2);

            rnn::pred_nn_t pred_nn2 = rnn::make_pred_nn(
                comp_graph2, l_args2.pred_param, nn2.layer.back().output);

            auto order2 = autodiff::topo_order(pred_nn2.logprob);
            autodiff::eval(order2, autodiff::eval_funcs);

            std::vector<std::vector<double>> inputs2;
            for (auto& o: pred_nn2.logprob) {
                auto& f = autodiff::get_output<la::vector<double>>(o);
                inputs2.push_back(std::vector<double> {f.data(), f.data() + f.size()});
            }

            s2.frames = inputs2;
            s2.ground_truth_fst = s.ground_truth_fst;

            scrf::e2e::make_graph(s2, l_args2);
            scrf::e2e::make_min_cost_gold(s2, l_args2);
            s2.cost = std::make_shared<scrf::seg_cost<ilat::fst>>(
                scrf::make_overlap_cost<ilat::fst>(*s2.ground_truth->fst, l_args2.sils));

            scrf::e2e::parameterize(s2, l_args2);

            scrf::composite_weight<ilat::fst>& graph_weight_func
                = *dynamic_cast<scrf::composite_weight<ilat::fst>*>(s2.graph.weight_func.get());
            graph_weight_func.weights.push_back(s2.cost);

            using hinge_loss = scrf::hinge_loss_with_frame_grad<
                scrf::e2e::iscrf, scrf::dense_vec,
                scrf::e2e::iscrf_path_maker>;

            hinge_loss loss_func2 { *s2.gold, s2.graph };

            std::cout << "numerical grad: " << (loss_func2.loss() - loss_func->loss()) / 1e-8 << std::endl;
        }
#endif

        if (ell < 0) {
            std::cout << "loss is less than zero.  skipping." << std::endl;
        }

        std::cout << std::endl;

        if (ebt::in(std::string("decay"), args)) {
            scrf::rmsprop_update(l_args.param, param_grad, l_args.opt_data,
                l_args.decay, l_args.step_size);
            lstm::rmsprop_update(l_args.nn_param, nn_param_grad, l_args.nn_opt_data,
                l_args.decay, l_args.step_size);
            rnn::rmsprop_update(l_args.pred_param, pred_grad, l_args.pred_opt_data,
                l_args.decay, l_args.step_size);
        } else {
            scrf::adagrad_update(l_args.param, param_grad, l_args.opt_data,
                l_args.step_size);
            lstm::adagrad_update(l_args.nn_param, nn_param_grad, l_args.nn_opt_data,
                l_args.step_size);
            rnn::adagrad_update(l_args.pred_param, pred_grad, l_args.pred_opt_data,
                l_args.step_size);
        }

        if (i % save_every == 0) {
            scrf::save_vec(l_args.param, "param-last");
            scrf::save_vec(l_args.opt_data, "opt-data-last");
            scrf::e2e::save_lstm_param(l_args.nn_param, l_args.pred_param, "nn-param-last");
            scrf::e2e::save_lstm_param(l_args.nn_opt_data, l_args.pred_opt_data, "nn-opt-data-last");
        }

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        ++i;
    }

    scrf::save_vec(l_args.param, output_param);
    scrf::save_vec(l_args.opt_data, output_opt_data);
    scrf::e2e::save_lstm_param(l_args.nn_param, l_args.pred_param, output_nn_param);
    scrf::e2e::save_lstm_param(l_args.nn_opt_data, l_args.pred_opt_data, output_nn_opt_data);

}

