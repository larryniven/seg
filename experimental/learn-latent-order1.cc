#include "scrf/experimental/iscrf.h"
#include "scrf/experimental/loss.h"
#include "scrf/experimental/scrf_weight.h"
#include "scrf/experimental/align.h"
#include <fstream>

struct learning_env {

    std::ifstream frame_batch;
    std::ifstream label_batch;

    std::ifstream lattice_batch;

    scrf::dense_vec align_param;

    int save_every;
    int update_align_every;

    std::string output_param;
    std::string output_opt_data;

    scrf::learning_args l_args;

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
            {"label-batch", "", true},
            {"lattice-batch", "", false},
            {"min-seg", "", false},
            {"max-seg", "", false},
            {"param", "", true},
            {"align-param", "", true},
            {"opt-data", "", true},
            {"step-size", "", true},
            {"momentum", "", false},
            {"features", "", true},
            {"save-every", "", false},
            {"update-align-every", "", false},
            {"output-param", "", false},
            {"output-opt-data", "", false},
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

    align_param = scrf::load_dense_vec(args.at("align-param"));

    scrf::parse_learning_args(l_args, args);
}

void learning_env::run()
{
    ebt::Timer timer;

    int i = 1;

    while (1) {

        scrf::learning_sample s { l_args };

        s.frames = speech::load_frame_batch(frame_batch);

        std::vector<std::string> label_seq = scrf::load_label_seq(label_batch);

        if (!label_batch) {
            break;
        }

        if (ebt::in(std::string("lattice-batch"), args)) {
            ilat::fst lat = ilat::load_lattice(lattice_batch, l_args.label_id);

            if (!lattice_batch) {
                std::cerr << "error reading " << args.at("lattice-batch") << std::endl;
                exit(1);
            }

            scrf::make_lattice(lat, s, l_args);
        } else {
            scrf::make_graph(s, l_args);
        }

        make_alignment_gold(align_param, label_seq, s, l_args);

        parameterize(s, l_args);
        s.cost = std::make_shared<scrf::mul<ilat::fst>>(scrf::mul<ilat::fst>(
            std::make_shared<scrf::seg_cost<ilat::fst>>(
                scrf::make_overlap_cost<ilat::fst>(s.gold_segs, l_args.sils)),
            l_args.cost_scale));

        std::cout << "gold path: ";
        for (auto& e: s.gold->edges()) {
            std::cout << l_args.id_label[s.gold->output(e)] << " ("
                << s.gold->time(s.gold->head(e)) << ") ";
        }
        std::cout << std::endl;
    
        std::shared_ptr<scrf::loss_func<scrf::dense_vec>> loss_func;

        if (args.at("loss") == "hinge-loss") {
            scrf::composite_weight<ilat::fst>& graph_weight_func
                = *dynamic_cast<scrf::composite_weight<ilat::fst>*>(s.graph.weight_func.get());
            graph_weight_func.weights.push_back(s.cost);

            using hinge_loss = scrf::hinge_loss<
                scrf::iscrf, scrf::dense_vec,
                scrf::iscrf_path_maker>;

            loss_func = std::make_shared<hinge_loss>(hinge_loss { *s.gold, s.graph });

            hinge_loss const& loss = *dynamic_cast<hinge_loss*>(loss_func.get());

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

        if (ell > 0) {
            param_grad = loss_func->param_grad();
        }

        if (ell < 0) {
            std::cout << "loss is less than zero.  skipping." << std::endl;
        }

        scrf::adagrad_update(l_args.param, param_grad, l_args.opt_data, l_args.step_size);

        if (i % save_every == 0) {
            scrf::save_vec(l_args.param, "param-last");
            scrf::save_vec(l_args.opt_data, "opt-data-last");
            scrf::save_vec(align_param, "align-param-last");
        }

        if (i % update_align_every == 0) {
            align_param = l_args.param;

            std::cout << std::endl;
            std::cout << "update align param" << std::endl;
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

}
