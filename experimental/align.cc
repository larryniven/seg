#include "scrf/experimental/align.h"
#include "scrf/experimental/scrf_feat.h"
#include "scrf/experimental/iscrf.h"
#include "scrf/experimental/segcost.h"
#include "scrf/experimental/scrf.h"

namespace iscrf {

    std::vector<std::string> load_label_seq(std::istream& is)
    {
        std::string line;
        std::getline(is, line);

        std::vector<std::string> parts;

        if (is) {
            parts = ebt::split(line);
            parts.pop_back();
        }

        return parts;
    }

    ilat::fst make_label_seq_fst(std::vector<std::string> const& label_seq,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label)
    {
        ilat::fst_data data;

        data.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(label_id);
        data.id_symbol = std::make_shared<std::vector<std::string>>(id_label);

        ilat::add_vertex(data, 0, ilat::vertex_data { 0 });

        for (int i = 0; i < label_seq.size(); ++i) {
            std::string const& s = label_seq.at(i);
            add_vertex(data, i + 1, ilat::vertex_data { 0 });
            add_edge(data, i, ilat::edge_data { i, i+1, 0, label_id.at(s), label_id.at(s) });
        }

        data.initials.push_back(0);
        data.finals.push_back(label_seq.size());

        ilat::fst result;
        result.data = std::make_shared<ilat::fst_data>(std::move(data));

        return result;
    }

    void make_alignment_gold(
        scrf::dense_vec const& ali_param,
        std::vector<std::string> const& label_seq,
        learning_sample& s,
        learning_args const& l_args)
    {
        ilat::fst label_seq_fst = make_label_seq_fst(label_seq,
            l_args.label_id, l_args.id_label);

        ilat::lazy_pair_mode2 pair_fst { *s.graph_data.fst, label_seq_fst };

        second_order::pair_scrf_data<scrf::dense_vec> ali_scrf_data;

        ali_scrf_data.fst = std::make_shared<ilat::lazy_pair_mode2>(pair_fst);
        ali_scrf_data.topo_order = std::make_shared<std::vector<std::tuple<int, int>>>(fst::topo_order(pair_fst));

        scrf::feat_dim_alloc alloc { l_args.labels };

        scrf::composite_feature<ilat::pair_fst, scrf::dense_vec> feat_func
            = second_order::make_feat<scrf::dense_vec, second_order::dense::pair_fst_lexicalizer>(
                alloc, l_args.features, s.frames, l_args.args);

        ali_scrf_data.feature_func = std::make_shared<scrf::composite_feature<ilat::pair_fst, scrf::dense_vec>>(feat_func);

        scrf::composite_weight<ilat::pair_fst> weight_func;

        weight_func.weights.push_back(
            std::make_shared<scrf::linear_score<ilat::pair_fst, scrf::dense_vec>>(
                scrf::linear_score<ilat::pair_fst, scrf::dense_vec> {
                    ali_param, ali_scrf_data.feature_func }));

        ali_scrf_data.weight_func = std::make_shared<scrf::composite_weight<ilat::pair_fst>>(weight_func);

        std::shared_ptr<ilat::pair_fst> path = scrf::shortest_path(ali_scrf_data);

        iscrf::iscrf_fst graph { s.graph_data };
        iscrf::second_order::pair_scrf_fst<scrf::dense_vec> ali_scrf { ali_scrf_data };

        std::vector<int> edges;

        for (auto& e: path->edges()) {
            edges.push_back(std::get<0>(e));

            s.gold_segs.push_back(segcost::segment<int> { graph.time(graph.tail(std::get<0>(e))),
                graph.time(graph.head(std::get<0>(e))), ali_scrf.output(e) });
        }

        s.gold_data.fst = ilat::ilat_path_maker()(edges, *s.graph_data.fst);
    }

    void make_even_gold(
        std::vector<std::string> const& label_seq,
        learning_sample& s,
        learning_args const& l_args)
    {
        int avg_dur = int(s.frames.size() / label_seq.size());

        int d = 0;
        for (int i = 0; i < label_seq.size(); ++i) {
            if (i != label_seq.size() - 1) {
                s.gold_segs.push_back(segcost::segment<int> {
                    d, d + avg_dur, l_args.label_id.at(label_seq.at(i)) });

                d += avg_dur;
            } else {
                s.gold_segs.push_back(segcost::segment<int> {
                    d, int(s.frames.size()), l_args.label_id.at(label_seq.at(i)) });
            }
        }

        iscrf::make_min_cost_gold(s, l_args);
    }

}
