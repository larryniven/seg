#include "scrf/experimental/iscrf_segnn.h"
#include "scrf/experimental/loss.h"
#include "scrf/experimental/scrf_weight.h"
#include "autodiff/autodiff.h"
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

    iscrf::segnn::learning_args l_args;

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

    iscrf::segnn::parse_learning_args(l_args, args);
}

void learning_env::run()
{
    int i = 1;

    while (1) {

        iscrf::learning_sample s { l_args };

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        s.gold_segs = iscrf::load_segments(gold_batch, l_args.label_id);

        if (!gold_batch) {
            break;
        }

        s.frames = frames;

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

        autodiff::computation_graph comp_graph;
        l_args.nn = residual::make_nn(comp_graph, l_args.nn_param);

        iscrf::make_min_cost_gold(s, l_args);

        iscrf::segnn::parameterize(s, l_args);

        double gold_cost = 0;

        iscrf::iscrf_fst gold { s.gold_data };
    
        std::cout << "gold path: ";
        for (auto& e: gold.edges()) {
            std::cout << l_args.id_label[gold.output(e)] << " ";
            gold_cost += cost(s.gold_data, e);
        }
        std::cout << std::endl;
    
        std::cout << "gold cost: " << gold_cost << std::endl;

        std::shared_ptr<scrf::loss_func_with_frame_grad<scrf::dense_vec, ilat::fst>> loss_func;

        if (args.at("loss") == "hinge-loss") {
            scrf::composite_weight<ilat::fst> weight_func_with_cost;
            weight_func_with_cost.weights.push_back(s.graph_data.weight_func);
            weight_func_with_cost.weights.push_back(s.graph_data.cost_func);
            s.graph_data.weight_func = std::make_shared<scrf::composite_weight<ilat::fst>>(weight_func_with_cost);

            using hinge_loss = scrf::hinge_loss<iscrf::iscrf_data>;

            loss_func = std::make_shared<hinge_loss>(
                hinge_loss { s.gold_data, s.graph_data });

            hinge_loss const& loss
                = *dynamic_cast<hinge_loss*>(loss_func.get());

            double gold_weight = 0;

            iscrf::iscrf_fst gold { s.gold_data };

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

            iscrf::segnn::learning_args l_args2;
            ::iscrf::parse_learning_args(l_args2, args);
            
            l_args2.nn_param = l_args.nn_param;
            l_args2.param = l_args.param;
            l_args2.nn_param.input_bias(0) += 1e-8;

            autodiff::computation_graph comp_graph2;
            l_args2.nn = residual::make_nn(comp_graph2, l_args2.nn_param);

            iscrf::learning_sample s2 { l_args2 };
            s2.gold_segs = s.gold_segs;
            s2.frames = s.frames;
            iscrf::make_graph(s2, l_args2);
            iscrf::make_min_cost_gold(s2, l_args2);
            iscrf::segnn::parameterize(s2, l_args2);

            scrf::composite_weight<ilat::fst> weight_func_with_cost;
            weight_func_with_cost.weights.push_back(s2.graph_data.weight_func);
            weight_func_with_cost.weights.push_back(s2.graph_data.cost_func);
            s2.graph_data.weight_func = std::make_shared<scrf::composite_weight<ilat::fst>>(weight_func_with_cost);

            using hinge_loss = scrf::hinge_loss<iscrf::iscrf_data>;

            hinge_loss loss_func2 { s2.gold_data, s2.graph_data };

            /*
            iscrf::iscrf_fst gold { s2.gold_data };

            double gold_weight = 0;

            std::cout << "gold: ";
            for (auto& e: gold.edges()) {
                std::cout << e << " " << l_args2.id_label[gold.output(e)] << " " << gold.weight(e) << std::endl;
                gold_weight += gold.weight(e);

                scrf::dense_vec f;
                (*s2.gold_data.feature_func)(f, *s2.gold_data.fst, e);
                auto& v = f.class_vec[0];
                std::cout << std::vector<double> { v.data(), v.data() + v.size() }  << std::endl;

                auto ch0 = autodiff::get_child(autodiff::get_child(l_args2.nn.layer[0].cell, 0), 0);
                auto& v_ch0 = autodiff::get_output<la::matrix<double>>(ch0);

                std::cout << ch0->name << std::endl;
                std::cout << v_ch0.cols() << std::endl;

                for (int i = 0; i < v_ch0.cols(); ++i) {
                    std::cout << v_ch0(0, i) << " ";
                }
                std::cout << std::endl;

                {
                    scrf::dense_vec f;
                    (*s.gold_data.feature_func)(f, *s.gold_data.fst, e);
                    auto& v = f.class_vec[0];
                    std::cout << std::vector<double> { v.data(), v.data() + v.size() }  << std::endl;

                    auto ch0 = autodiff::get_child(autodiff::get_child(l_args.nn.layer[0].cell, 0), 0);
                    auto& v_ch0 = autodiff::get_output<la::matrix<double>>(ch0);

                    std::cout << ch0->name << std::endl;
                    std::cout << v_ch0.cols() << std::endl;

                    for (int i = 0; i < v_ch0.cols(); ++i) {
                        std::cout << v_ch0(0, i) << " ";
                    }
                    std::cout << std::endl;
                }
            }
            std::cout << std::endl;

            std::cout << "gold score: " << gold_weight << std::endl;

            double graph_weight = 0;

            iscrf::iscrf_fst graph_path { loss_func2.graph_path };

            std::cout << "cost aug: ";
            for (auto& e: graph_path.edges()) {
                std::cout << l_args2.id_label[graph_path.output(e)] << " ";
                graph_weight += graph_path.weight(e);
            }
            std::cout << std::endl;

            std::cout << "cost aug score: " << graph_weight << std::endl;
            */

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
        residual::nn_param_t nn_param_grad;

        residual::resize_as(nn_param_grad, l_args.nn_param);

        if (ell > 0) {
            param_grad = loss_func->param_grad();

            // compute gradient

            using hinge_loss = scrf::hinge_loss<iscrf::iscrf_data>;

            hinge_loss const& loss
                = *dynamic_cast<hinge_loss*>(loss_func.get());

            iscrf::iscrf_fst gold { s.gold_data };

            scrf::dense_vec neg_param = l_args.param;
            scrf::imul(neg_param, -1);

            std::shared_ptr<segnn::segnn_feat> segnn_feat = std::dynamic_pointer_cast<segnn::segnn_feat>(
                s.gold_data.feature_func->features[0]);

            residual::nn_param_t zero;
            residual::resize_as(zero, l_args.nn_param);

            for (auto& e: gold.edges()) {
                segnn_feat->gradient = zero;

                segnn_feat->grad(neg_param, *s.gold_data.fst, e);

                residual::iadd(nn_param_grad, segnn_feat->gradient);
            }

            segnn_feat = std::dynamic_pointer_cast<segnn::segnn_feat>(
                s.graph_data.feature_func->features[0]);

            iscrf::iscrf_fst graph_path { loss.graph_path };

            for (auto& e: graph_path.edges()) {
                segnn_feat->gradient = zero;

                segnn_feat->grad(l_args.param, *s.graph_data.fst, e);

                residual::iadd(nn_param_grad, segnn_feat->gradient);
            }

            std::cout << "analytical grad: " << nn_param_grad.input_bias(0) << std::endl;

            scrf::adagrad_update(l_args.param, param_grad, l_args.opt_data,
                l_args.step_size);
            residual::adagrad_update(l_args.nn_param, nn_param_grad, l_args.nn_opt_data,
                l_args.step_size);

        }

        if (ell < 0) {
            std::cout << "loss is less than zero.  skipping." << std::endl;
        }

        std::cout << std::endl;

        if (i % save_every == 0) {
            scrf::save_vec(l_args.param, "param-last");
            scrf::save_vec(l_args.opt_data, "opt-data-last");
            residual::save_nn_param(l_args.nn_param, "nn-param-last");
            residual::save_nn_param(l_args.nn_opt_data, "nn-opt-data-last");
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
    residual::save_nn_param(l_args.nn_param, output_nn_param);
    residual::save_nn_param(l_args.nn_opt_data, output_nn_opt_data);

}

