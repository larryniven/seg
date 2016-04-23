#include "scrf/experimental/pair_scrf.h"
#include "scrf/experimental/scrf_weight.h"

namespace scrf {

    namespace second_order {

        namespace sparse {

            double backoff_cost::operator()(ilat::pair_fst const& f,
                ilat::pair_fst::edge e) const
            {
                return (f.fst1().output(std::get<0>(e)) == 0 ? 1 : 0);
            }

            sample::sample(inference_args const& args)
                : graph_alloc(args.labels)
            {
            }
            
            void make_lattice(ilat::fst const& lat, sample& s, inference_args const& i_args)
            {
                ilat::fst lat_with_loops;
                lat_with_loops.data = std::make_shared<ilat::fst_data>(*lat.data);
                add_eps_loops(lat_with_loops);

                s.graph.fst = std::make_shared<ilat::lazy_pair>(ilat::lazy_pair { lat_with_loops, i_args.lm });

                auto lat_order = fst::topo_order(lat);
                for (auto& v: lat_order) {
                    for (int i = i_args.lm.vertices().size() - 1; i >= 0; --i) {
                        s.graph.topo_order_cache.push_back(std::make_tuple(v, i));
                    }
                }

                composite_feature<ilat::pair_fst, sparse_vec> graph_feat_func
                    = make_feat<sparse_vec, pair_fst_lexicalizer<sparse_vec>>(s.graph_alloc, i_args.features, s.frames, i_args.args);
            
                composite_weight<ilat::pair_fst> weight;
                weight.weights.push_back(std::make_shared<linear_score<ilat::pair_fst, sparse_vec>>(
                    linear_score<ilat::pair_fst, sparse_vec>(i_args.param,
                        std::make_shared<composite_feature<ilat::pair_fst, sparse_vec>>(graph_feat_func))));
            
                s.graph.weight_func = std::make_shared<composite_weight<ilat::pair_fst>>(weight);
                s.graph.feature_func = std::make_shared<composite_feature<ilat::pair_fst, sparse_vec>>(graph_feat_func);
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

                l_args.param = load_sparse_vec(args.at("param"));
                l_args.opt_data = load_sparse_vec(args.at("opt-data"));
                l_args.step_size = std::stod(args.at("step-size"));
            
                l_args.momentum = -1;
                if (ebt::in(std::string("momentum"), args)) {
                    l_args.momentum = std::stod(args.at("momentum"));
                    assert(0 <= l_args.momentum && l_args.momentum <= 1);
                }
            
                l_args.features = ebt::split(args.at("features"), ",");
            
                l_args.label_id = load_label_id(args.at("label"));
            
                l_args.id_label.resize(l_args.label_id.size());
                for (auto& p: l_args.label_id) {
                    l_args.labels.push_back(p.second);
                    l_args.id_label[p.second] = p.first;
                }
            
                l_args.sils.push_back(l_args.label_id.at("<s>"));
                l_args.sils.push_back(l_args.label_id.at("</s>"));
                l_args.sils.push_back(l_args.label_id.at("sil"));
            
                l_args.lm = ilat::load_arpa_lm(l_args.args.at("lm"), l_args.label_id);

                return l_args;
            }
            
            learning_sample::learning_sample(learning_args const& args)
                : sample(args), gold_alloc(args.labels)
            {
            }
            
            void make_min_cost_gold(learning_sample& s, learning_args const& l_args)
            {
                auto old_weight_func = s.graph.weight_func;

                s.graph.weight_func = std::make_shared<mul<ilat::pair_fst>>(mul<ilat::pair_fst>(
                    std::make_shared<seg_cost<ilat::pair_fst>>(
                        make_overlap_cost<ilat::pair_fst>(s.gold_segs, l_args.sils)),
                    -1));

                s.gold = fst::shortest_path<pair_scrf<sparse_vec>, pair_scrf_path_maker<sparse_vec>>(s.graph);

                s.graph.weight_func = old_weight_func;
            
                using comp_feat = composite_feature<ilat::pair_fst, sparse_vec>;

                comp_feat gold_feat_func = make_feat<sparse_vec, pair_fst_lexicalizer<sparse_vec>>(s.gold_alloc, l_args.features, s.frames, l_args.args);
            
                s.gold->weight_func = std::make_shared<linear_score<ilat::pair_fst, sparse_vec>>(
                    linear_score<ilat::pair_fst, sparse_vec>(l_args.param,
                    std::make_shared<comp_feat>(gold_feat_func)));
                s.gold->feature_func = std::make_shared<comp_feat>(gold_feat_func);
            }

        }

        namespace dense {

        }

    }

}
