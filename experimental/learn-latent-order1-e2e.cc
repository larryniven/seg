#include "scrf/experimental/iscrf_e2e.h"
#include "scrf/experimental/loss.h"
#include "scrf/experimental/scrf_weight.h"
#include "scrf/experimental/align.h"
#include "nn/lstm.h"
#include <random>
#include <fstream>

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream label_batch;

    std::ifstream lattice_batch;

    scrf::dense_vec align_param;
    lstm::dblstm_feat_param_t align_nn_param;
    nn::pred_param_t align_pred_param;

    int save_every;
    int update_align_every;

    std::string output_param;
    std::string output_opt_data;

    std::string output_nn_param;
    std::string output_nn_opt_data;

    iscrf::e2e::learning_args l_args;

    double grad_noise_var;
    double grad_noise_time;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

void gaussian(la::vector<double>& v, double var, std::default_random_engine& gen);
void gaussian(la::matrix<double>& m, double var, std::default_random_engine& gen);
void gaussian(lstm::lstm_unit_param_t& param, double var, std::default_random_engine& gen);
void gaussian(lstm::blstm_feat_param_t& param, double var, std::default_random_engine& gen);
void gaussian(lstm::dblstm_feat_param_t& param, double var, std::default_random_engine& gen);

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "learn-first",
        "Learn segmental CRF",
        {
            {"frame-batch", "", true},
            {"label-batch", "", true},
            {"lattice-batch", "", false},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"l2", "", false},
            {"param", "", true},
            {"opt-data", "", true},
            {"nn-param", "", true},
            {"nn-opt-data", "", true},
            {"align-param", "", true},
            {"align-nn-param", "", true},
            {"step-size", "", true},
            {"decay", "", false},
            {"features", "", true},
            {"save-every", "", false},
            {"update-align-every", "", false},
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
            {"frame-softmax", "", false},
            {"even-init", "", false},
            {"freeze-encoder", "", false},
            {"grad-noise-var", "", false},
            {"grad-noise-time", "", false}
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

    label_batch.open(args.at("label-batch"));

    if (ebt::in(std::string("lattice-batch"), args)) {
        lattice_batch.open(args.at("lattice-batch"));
    }

    save_every = std::numeric_limits<int>::max();
    if (ebt::in(std::string("save-every"), args)) {
        save_every = std::stoi(args.at("save-every"));
    }

    update_align_every = std::numeric_limits<int>::max();
    if (ebt::in(std::string("update-align-every"), args)) {
        update_align_every = std::stoi(args.at("update-align-every"));
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

    iscrf::e2e::parse_learning_args(l_args, args);

    align_param = scrf::load_dense_vec(args.at("align-param"));
    std::tie(align_nn_param, align_pred_param) = iscrf::e2e::load_lstm_param(args.at("align-nn-param"));

    grad_noise_var = 0;
    if (ebt::in(std::string("grad-noise-var"), args)) {
        grad_noise_var = std::stod(args.at("grad-noise-var"));
    }

    grad_noise_time = 0;
    if (ebt::in(std::string("grad-noise-time"), args)) {
        grad_noise_time = std::stod(args.at("grad-noise-time"));
    }
}

void learning_env::run()
{
    ebt::Timer timer;

    int i = 1;

    std::default_random_engine gen { l_args.rnndrop_seed };

    while (1) {

        iscrf::learning_sample s { l_args };

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        std::vector<std::string> label_seq = iscrf::load_label_seq(label_batch);

        if (!label_batch || !frame_batch) {
            break;
        }

        autodiff::computation_graph comp_graph;
        lstm::dblstm_feat_nn_t nn;
        rnn::pred_nn_t pred_nn;

        std::vector<std::shared_ptr<autodiff::op_t>> upsampled_output
            = iscrf::e2e::make_input(comp_graph, nn, pred_nn, frames, gen, l_args);

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

            iscrf::make_lattice(lat, s, l_args);
        } else {
            iscrf::make_graph(s, l_args);
        }

        if (ebt::in(std::string("even-init"), args) && i <= update_align_every) {
            iscrf::make_even_gold(label_seq, s, l_args);
        } else {
            iscrf::make_alignment_gold(align_param, label_seq, s, l_args);
        }

        iscrf::parameterize(s, l_args);

        iscrf::iscrf_fst gold { s.gold_data };

        std::cout << "gold path: ";
        for (auto& e: gold.edges()) {
            std::cout << l_args.id_label[gold.output(e)] << " ("
                << gold.time(gold.head(e)) << ") ";
        }
        std::cout << std::endl;
    
        std::shared_ptr<scrf::loss_func_with_frame_grad<scrf::dense_vec, ilat::fst>> loss_func;

        if (args.at("loss") == "hinge-loss") {
            scrf::composite_weight<ilat::fst> weight_func_with_cost;
            weight_func_with_cost.weights.push_back(s.graph_data.weight_func);
            weight_func_with_cost.weights.push_back(s.graph_data.cost_func);
            s.graph_data.weight_func = std::make_shared<scrf::composite_weight<ilat::fst>>(weight_func_with_cost);

            using hinge_loss = scrf::hinge_loss<iscrf::iscrf_data>;

            loss_func = std::make_shared<hinge_loss>(hinge_loss { s.gold_data, s.graph_data });

            hinge_loss const& loss = *dynamic_cast<hinge_loss*>(loss_func.get());

            double gold_weight = 0;

            std::cout << "gold: ";
            for (auto& e: gold.edges()) {
                std::cout << l_args.id_label[gold.output(e)] << " " << gold.weight(e) << " ";
                gold_weight += gold.weight(e);
            }
            std::cout << std::endl;

            std::cout << "gold score: " << gold_weight << std::endl;

            double graph_weight = 0;

            iscrf::iscrf_fst graph_path { loss.graph_path };

            std::cout << "cost aug: ";
            for (auto& e: graph_path.edges()) {
                std::cout << l_args.id_label[graph_path.output(e)] << " " << graph_path.weight(e) << " ";
                graph_weight += graph_path.weight(e);
            }
            std::cout << std::endl;

            std::cout << "cost aug score: " << graph_weight << std::endl;
        } else {
            std::cout << "unknown loss function " << args.at("loss") << std::endl;
            exit(1);
        }

#if 0
        {
            double ell1 = loss_func->loss();

            iscrf::e2e::learning_args l_args2 = l_args;
            l_args2.nn_param.layer.back().forward_output_weight(0, 0) += 1e-8;

            iscrf::learning_sample s2 { l_args2 };

            autodiff::computation_graph comp_graph2;
            lstm::dblstm_feat_nn_t nn2;
            rnn::pred_nn_t pred_nn2;

            std::vector<std::shared_ptr<autodiff::op_t>> upsampled_output2
                = iscrf::e2e::make_input(comp_graph2, nn2, pred_nn2, frames, gen, l_args2);

            auto order2 = autodiff::topo_order(upsampled_output2);
            autodiff::eval(order2, autodiff::eval_funcs);

            std::vector<std::vector<double>> inputs2;
            for (auto& o: upsampled_output2) {
                auto& f = autodiff::get_output<la::vector<double>>(o);
                inputs2.push_back(std::vector<double> {f.data(), f.data() + f.size()});
            }

            s2.frames = inputs2;

            iscrf::make_graph(s2, l_args2);
            iscrf::make_alignment_gold(align_param, label_seq, s2, l_args2);
            iscrf::parameterize(s2, l_args2);

            scrf::composite_weight<ilat::fst> weight_func_with_cost;
            weight_func_with_cost.weights.push_back(s2.graph_data.weight_func);
            weight_func_with_cost.weights.push_back(s2.graph_data.cost_func);
            s2.graph_data.weight_func = std::make_shared<scrf::composite_weight<ilat::fst>>(weight_func_with_cost);

            using hinge_loss = scrf::hinge_loss<iscrf::iscrf_data>;

            hinge_loss loss_func2 { s2.gold_data, s2.graph_data };

            double ell2 = loss_func2.loss();

            std::cout << "loss1: " << ell1 << " loss2: " << ell2 << std::endl;
            std::cout << "num grad: " << (ell2 - ell1) / 1e-8 << std::endl;

        }
#endif

        std::cout << "gold segs: " << s.gold_data.fst->edges().size()
            << " frames: " << s.frames.size() << std::endl;

        double ell = loss_func->loss();

        std::cout << "loss: " << ell << std::endl;

        scrf::dense_vec param_grad;
        lstm::dblstm_feat_param_t nn_param_grad;
        nn::pred_param_t pred_grad;

        if (ell > 0) {
            param_grad = loss_func->param_grad();

            if (!ebt::in(std::string("freeze-encoder"), l_args.args)) {
                std::vector<std::vector<double>> frame_grad;
                frame_grad.resize(inputs.size());
                for (int i = 0; i < inputs.size(); ++i) {
                    frame_grad[i].resize(inputs[i].size());
                }

                std::shared_ptr<scrf::composite_feature_with_frame_grad<ilat::fst, scrf::dense_vec>> feat_func
                    = iscrf::e2e::filter_feat_with_frame_grad(s.graph_data);

                loss_func->frame_grad(*feat_func, frame_grad, l_args.param);

                for (int i = 0; i < frame_grad.size(); ++i) {
                    upsampled_output[i]->grad = std::make_shared<la::vector<double>>(
                        frame_grad[i]);
                }

                autodiff::grad(order, autodiff::grad_funcs);

                nn_param_grad = lstm::copy_dblstm_feat_grad(nn);

                if (ebt::in(std::string("frame-softmax"), args)) {
                    pred_grad = rnn::copy_grad(pred_nn);

                }

                std::cout << "analytical grad: " << nn_param_grad.layer.back().forward_output_weight(0, 0) << std::endl;

                if (ebt::in(std::string("l2"), l_args.args)) {
                    scrf::dense_vec p = l_args.param;
                    scrf::imul(p, l_args.l2);
                    scrf::iadd(param_grad, p);

                    lstm::dblstm_feat_param_t nn_p = l_args.nn_param;
                    lstm::imul(nn_p, l_args.l2);
                    lstm::iadd(nn_param_grad, nn_p);
                }

                if (ebt::in(std::string("grad-noise-var"), l_args.args)) {
                    lstm::dblstm_feat_param_t noise = l_args.nn_param;
                    gaussian(noise, grad_noise_var / std::sqrt(1 + grad_noise_time), gen);
                    lstm::iadd(nn_param_grad, noise);
                }

            }

            double v1 = l_args.nn_param.layer.front().forward_param.hidden_output(0, 0);

            if (ebt::in(std::string("decay"), args)) {
                scrf::rmsprop_update(l_args.param, param_grad, l_args.opt_data,
                    l_args.decay, l_args.step_size);

                if (!ebt::in(std::string("freeze-encoder"), l_args.args)) {
                    lstm::rmsprop_update(l_args.nn_param, nn_param_grad, l_args.nn_opt_data,
                        l_args.decay, l_args.step_size);

                    if (ebt::in(std::string("frame-softmax"), args)) {
                        nn::rmsprop_update(l_args.pred_param, pred_grad, l_args.pred_opt_data,
                            l_args.decay, l_args.step_size);
                    }
                }
            } else {
                scrf::adagrad_update(l_args.param, param_grad, l_args.opt_data,
                    l_args.step_size);

                if (!ebt::in(std::string("freeze-encoder"), l_args.args)) {
                    lstm::adagrad_update(l_args.nn_param, nn_param_grad, l_args.nn_opt_data,
                        l_args.step_size);

                    if (ebt::in(std::string("frame-softmax"), args)) {
                        nn::adagrad_update(l_args.pred_param, pred_grad, l_args.pred_opt_data,
                            l_args.step_size);
                    }
                }
            }

            double v2 = l_args.nn_param.layer.front().forward_param.hidden_output(0, 0);

            std::cout << "weight: " << v1 << " update: " << v2 - v1 << " rate: " << (v2 - v1) / v1 << std::endl;

            if (i % save_every == 0) {
                scrf::save_vec(l_args.param, "param-last");
                scrf::save_vec(l_args.opt_data, "opt-data-last");
                iscrf::e2e::save_lstm_param(l_args.nn_param, l_args.pred_param, "nn-param-last");
                iscrf::e2e::save_lstm_param(l_args.nn_opt_data, l_args.pred_opt_data, "nn-opt-data-last");
            }

            if (i % update_align_every == 0) {
                align_param = l_args.param;

                if (!ebt::in(std::string("freeze-encoder"), l_args.args)) {
                    align_nn_param = l_args.nn_param;
                    align_pred_param = l_args.pred_param;
                }

                std::cout << std::endl;
                std::cout << "update align param" << std::endl;
            }

        }

        if (ell < 0) {
            std::cout << "loss is less than zero.  skipping." << std::endl;
        }

        std::cout << std::endl;

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        ++i;
    }

    scrf::save_vec(l_args.param, output_param);
    scrf::save_vec(l_args.opt_data, output_opt_data);
    iscrf::e2e::save_lstm_param(l_args.nn_param, l_args.pred_param, output_nn_param);
    iscrf::e2e::save_lstm_param(l_args.nn_opt_data, l_args.pred_opt_data, output_nn_opt_data);

}

void gaussian(la::vector<double>& v, double var, std::default_random_engine& gen)
{
    std::normal_distribution<double> dist { 0, std::sqrt(var) };

    for (int i = 0; i < v.size(); ++i) {
        v(i) = dist(gen);
    }
}

void gaussian(la::matrix<double>& m, double var, std::default_random_engine& gen)
{
    std::normal_distribution<double> dist { 0, std::sqrt(var) };

    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.cols(); ++j) {
            m(i, j) = dist(gen);
        }
    }
}

void gaussian(lstm::lstm_unit_param_t& param, double var, std::default_random_engine& gen)
{
    gaussian(param.hidden_input, var, gen);
    gaussian(param.hidden_output, var, gen);
    gaussian(param.hidden_bias, var, gen);

    gaussian(param.input_input, var, gen);
    gaussian(param.input_output, var, gen);
    gaussian(param.input_peep, var, gen);
    gaussian(param.input_bias, var, gen);

    gaussian(param.output_input, var, gen);
    gaussian(param.output_output, var, gen);
    gaussian(param.output_peep, var, gen);
    gaussian(param.output_bias, var, gen);

    gaussian(param.forget_input, var, gen);
    gaussian(param.forget_output, var, gen);
    gaussian(param.forget_peep, var, gen);
    gaussian(param.forget_bias, var, gen);
}

void gaussian(lstm::blstm_feat_param_t& param, double var, std::default_random_engine& gen)
{
    gaussian(param.forward_param, var, gen);
    gaussian(param.backward_param, var, gen);

    gaussian(param.forward_output_weight, var, gen);
    gaussian(param.backward_output_weight, var, gen);
    gaussian(param.output_bias, var, gen);
}

void gaussian(lstm::dblstm_feat_param_t& param, double var, std::default_random_engine& gen)
{
    for (auto& ell: param.layer) {
        gaussian(ell, var, gen);
    }
}
