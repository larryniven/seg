#include "scrf/experimental/align.h"
#include "scrf/experimental/scrf_feat.h"
#include "scrf/experimental/iscrf.h"
#include "scrf/experimental/segcost.h"
#include "scrf/experimental/scrf.h"

namespace scrf {

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
        dense_vec const& ali_param,
        std::vector<std::string> const& label_seq,
        learning_sample& s,
        learning_args const& i_args)
    {
        ilat::fst label_seq_fst = make_label_seq_fst(label_seq,
            i_args.label_id, i_args.id_label);

        ilat::lazy_pair pair_fst { *s.graph_data.fst, label_seq_fst};

        second_order::pair_scrf<dense_vec> ali_scrf;

        ali_scrf.fst = std::make_shared<ilat::lazy_pair>(pair_fst);
        ali_scrf.topo_order_cache = fst::topo_order(pair_fst);

        feat_dim_alloc alloc { i_args.labels };

        composite_feature<ilat::pair_fst, dense_vec> feat_func
            = second_order::make_feat<dense_vec, second_order::dense::pair_fst_lexicalizer>(
                alloc, i_args.features, s.frames, i_args.args);

        ali_scrf.feature_func = std::make_shared<composite_feature<ilat::pair_fst, dense_vec>>(feat_func);

        composite_weight<ilat::pair_fst> weight_func;

        weight_func.weights.push_back(
            std::make_shared<linear_score<ilat::pair_fst, dense_vec>>(
                linear_score<ilat::pair_fst, dense_vec> {
                    ali_param, ali_scrf.feature_func }));

        ali_scrf.weight_func = std::make_shared<composite_weight<ilat::pair_fst>>(weight_func);

        std::shared_ptr<second_order::pair_scrf<dense_vec>> path
            = fst::shortest_path<second_order::pair_scrf<dense_vec>,
                second_order::pair_scrf_path_maker<dense_vec>>(ali_scrf);

        std::vector<int> edges;

        for (auto& e: path->edges()) {
            edges.push_back(std::get<0>(e));

            s.gold_segs.push_back(segcost::segment<int> { s.graph.time(s.graph.tail(std::get<0>(e))),
                s.graph.time(s.graph.head(std::get<0>(e))), ali_scrf.output(e) });
        }

        s.gold = iscrf_path_maker()(edges, s.graph);
    }

}
