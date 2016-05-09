#include "scrf/experimental/iscrf_e2e.h"
#include "scrf/experimental/loss.h"
#include "scrf/experimental/scrf_weight.h"
#include "autodiff/autodiff.h"
#include "nn/lstm.h"
#include <fstream>

struct prediction_env {

    std::ifstream frame_batch;

    std::ifstream lattice_batch;

    iscrf::e2e::inference_args i_args;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "predict-order1-e2e",
        "Decode with segmental CRF",
        {
            {"frame-batch", "", true},
            {"lattice-batch", "", false},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"param", "", true},
            {"nn-param", "", true},
            {"features", "", true},
            {"label", "", true},
            {"rnndrop-prob", "", false},
            {"subsample-freq", "", false},
            {"subsample-shift", "", false},
            {"frame-softmax", "", false}
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

    prediction_env env { args };

    env.run();

    return 0;
}

prediction_env::prediction_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    if (ebt::in(std::string("frame-batch"), args)) {
        frame_batch.open(args.at("frame-batch"));
    }

    if (ebt::in(std::string("lattice-batch"), args)) {
        lattice_batch.open(args.at("lattice-batch"));
    }

    iscrf::e2e::parse_inference_args(i_args, args);
}

void prediction_env::run()
{
    int i = 1;

    while (1) {

        iscrf::sample s { i_args };

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        autodiff::computation_graph comp_graph;

        std::vector<std::shared_ptr<autodiff::op_t>> frame_ops;
        for (auto& f: frames) {
            frame_ops.push_back(comp_graph.var(la::vector<double>(f)));
        }

        std::vector<std::shared_ptr<autodiff::op_t>> subsampled_input;
        if (i_args.subsample_freq > 1) {
            subsampled_input = rnn::subsample_input(frame_ops,
                i_args.subsample_freq, i_args.subsample_shift);
        } else {
            subsampled_input = frame_ops;
        }

        lstm::dblstm_feat_nn_t nn = lstm::make_dblstm_feat_nn(comp_graph, i_args.nn_param, subsampled_input);

        if (ebt::in(std::string("rnndrop-prob"), args)) {
            lstm::apply_mask(nn, i_args.nn_param, i_args.rnndrop_prob);
        }

        rnn::pred_nn_t pred_nn;

        std::vector<std::shared_ptr<autodiff::op_t>> output;

        if (ebt::in(std::string("frame-softmax"), args)) {
            pred_nn = rnn::make_pred_nn(comp_graph, i_args.pred_param, nn.layer.back().output);

            output = pred_nn.logprob;
        } else {
            output = nn.layer.back().output;
        }

        std::vector<std::shared_ptr<autodiff::op_t>> upsampled_output;
        if (i_args.subsample_freq > 1) {
             upsampled_output = rnn::upsample_output(output,
                 i_args.subsample_freq, i_args.subsample_shift, frames.size());
        } else {
             upsampled_output = output;
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
            ilat::fst lat = ilat::load_lattice(lattice_batch, i_args.label_id);

            if (!lattice_batch) {
                std::cerr << "error reading " << args.at("lattice-batch") << std::endl;
                exit(1);
            }

            iscrf::make_lattice(lat, s, i_args);
        } else {
            iscrf::make_graph(s, i_args);
        }

        iscrf::parameterize(s.graph_data, s.graph_alloc, s.frames, i_args);

        std::shared_ptr<ilat::fst> graph_path = scrf::shortest_path<iscrf::iscrf_data>(s.graph_data);

        for (auto& e: graph_path->edges()) {
            std::cout << i_args.id_label.at(graph_path->output(e)) << " ";
        }
        std::cout << "(" << i << ".phn)" << std::endl;

#if DEBUG_TOP
        if (i == DEBUG_TOP) {
            break;
        }
#endif

        ++i;
    }

}

