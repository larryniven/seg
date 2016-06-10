#include "scrf/experimental/iscrf.h"
#include "scrf/experimental/loss.h"
#include "scrf/experimental/scrf_weight.h"
#include <fstream>

struct oracle_env {

    std::ifstream label_batch;

    std::ifstream lattice_batch;

    iscrf::inference_args i_args;

    std::unordered_map<std::string, std::string> args;

    oracle_env(std::unordered_map<std::string, std::string> args);

    void run();

};

ilat::fst make_label_fst(std::vector<std::string> const& label_seq,
    std::unordered_map<std::string, int> const& label_id);

std::tuple<int, int, int, int> error_analysis(std::vector<std::tuple<int, int>> const& edges,
    ilat::lazy_pair_mode1 const& composed_fst);

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "oracle-error",
        "Calculate orracle error rate of a lattice",
        {
            {"label-batch", "", true},
            {"lattice-batch", "", true},
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

    oracle_env env { args };

    env.run();

    return 0;
}

oracle_env::oracle_env(std::unordered_map<std::string, std::string> args)
{
    label_batch.open(args.at("label-batch"));

    lattice_batch.open(args.at("lattice-batch"));

    i_args.label_id = scrf::load_label_id(args.at("label"));

    i_args.id_label.resize(i_args.label_id.size());
    for (auto& p: i_args.label_id) {
        i_args.labels.push_back(p.second);
        i_args.id_label[p.second] = p.first;
    }
}

void oracle_env::run()
{
    
    ebt::Timer timer;

    int i = 1;

    int total_len = 0;

    int total_ins = 0;
    int total_del = 0;
    int total_sub = 0;

    while (1) {

        iscrf::sample s { i_args };

        std::vector<std::string> label_seq = iscrf::load_labels(label_batch);

        if (!label_batch) {
            break;
        }

        ilat::fst lat = ilat::load_lattice(lattice_batch, i_args.label_id);

        if (!lattice_batch) {
            break;
        }

        for (auto& e: lat.data->edges) {
            e.weight = 0;
        }

        iscrf::make_lattice(lat, s);

        ilat::add_eps_loops(lat);

        ilat::fst label_fst = make_label_fst(label_seq, i_args.label_id);

        ilat::lazy_pair_mode1 composed_fst { lat, label_fst };

        auto topo_order = fst::topo_order(composed_fst);

        fst::forward_one_best<ilat::lazy_pair_mode1> one_best;
        for (auto& i: composed_fst.initials()) {
            one_best.extra[i] = { std::make_tuple(-1, -1), 0 };
        }
        one_best.merge(composed_fst, topo_order);
        std::vector<std::tuple<int, int>> best_edges = one_best.best_path(composed_fst);

        int ins = 0;
        int del = 0;
        int sub = 0;
        int length = 0;

        std::tie(ins, del, sub, length) = error_analysis(best_edges, composed_fst);

        std::cout << i << ": edges: " << lat.edges().size() << " density: " << lat.edges().size() / length << std::endl;

        std::cout << "best ins: " << ins << " del: " << del << " sub: " << sub << " len: " << length
            << " er: " << double(ins + del + sub) / length << std::endl;

        total_ins += ins;
        total_del += del;
        total_sub += sub;
        total_len += length;

        ++i;
    }

    std::cout << "total ins: " << total_ins
        << " total del: " << total_del
        << " total sub: " << total_sub
        << " total len: " << total_len
        << " er: " << double(total_ins + total_del + total_sub) / total_len << std::endl;
}

ilat::fst make_label_fst(std::vector<std::string> const& label_seq,
    std::unordered_map<std::string, int> const& label_id)
{
    ilat::fst_data data;
    data.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(label_id);

    int v = 0;
    ilat::add_vertex(data, v, ilat::vertex_data { v });

    for (int i = 0; i < label_seq.size(); ++i) {
        int u = data.vertices.size();
        ilat::add_vertex(data, u, ilat::vertex_data { u });

        // substitution & insertion
        for (int ell = 0; ell < label_id.size(); ++ell) {
            int e = data.edges.size();
            ilat::add_edge(data, e, ilat::edge_data { v, u,
                (ell == label_id.at(label_seq.at(i)) ? 0.0 : -1.0),
                ell, label_id.at(label_seq.at(i)) });
        }

        v = u;
    }

    data.initials.push_back(0);
    data.finals.push_back(v);

    for (int v = 0; v < data.vertices.size(); ++v) {
        // deletion
        for (int ell = 1; ell < label_id.size(); ++ell) {
            int e = data.edges.size();
            ilat::add_edge(data, e, ilat::edge_data { v, v,
                -1, ell, 0 });
        }
    }

    ilat::fst f;
    f.data = std::make_shared<ilat::fst_data>(data);

    return f;
}

std::tuple<int, int, int, int> error_analysis(std::vector<std::tuple<int, int>> const& edges,
    ilat::lazy_pair_mode1 const& composed_fst)
{
    int ins = 0;
    int del = 0;
    int sub = 0;
    int length = 0;

    for (auto& e: edges) {
        if (composed_fst.input(e) == 0) {
            ++del;
        } else if (composed_fst.output(e) == 0) {
            ++ins;
        } else if (composed_fst.input(e) != composed_fst.output(e)) {
            ++sub;
        }

        if (composed_fst.output(e) != 0) {
            ++length;
        }
    }

    return std::make_tuple(ins, del, sub, length);

}


