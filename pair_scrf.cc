#include "seg/pair_scrf.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"

namespace iscrf {

    namespace second_order {

        namespace sparse {

            double* pair_fst_lexicalizer::lex(
                scrf::feat_dim_alloc const& alloc, int order,
                scrf::sparse_vec& feat, ilat::pair_fst const& fst,
                ilat::pair_fst::edge e) const
            {
                std::string label_tuple;
                auto const& vertex_attrs = fst.fst2().data->vertex_attrs;
                auto const& id_symbol = *fst.fst1().data->id_symbol;

                if (order == 0) {
                    // do nothing
                } else if (order == 1) {
                    label_tuple = id_symbol.at(fst.output(e));
                } else if (order == 2) {
                    for (auto& p: vertex_attrs.at(std::get<1>(fst.tail(e)))) {
                        if (p.first == "history") {
                            label_tuple = p.second;
                            break;
                        }
                    }

                    label_tuple += "_" + id_symbol.at(fst.output(e));
                } else {
                    std::cerr << "order " << order << " not implemented" << std::endl;
                    exit(1);
                }

                la::vector<double>& g = feat.class_vec[label_tuple];
                g.resize(alloc.order_dim[order]);

                return g.data();
            }

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

                s.graph_data.fst = std::make_shared<ilat::lazy_pair_mode1>(ilat::lazy_pair_mode1 { lat_with_loops, i_args.lm });

                auto lat_order = fst::topo_order(lat);
                for (auto& v: lat_order) {
                    for (int i = i_args.lm.vertices().size() - 1; i >= 0; --i) {
                        s.graph_data.topo_order->push_back(std::make_tuple(v, i));
                    }
                }

                scrf::composite_feature<ilat::pair_fst, scrf::sparse_vec> graph_feat_func
                    = make_feat<scrf::sparse_vec, pair_fst_lexicalizer>(
                        s.graph_alloc, i_args.features, s.frames, i_args.args);
            
                scrf::composite_weight<ilat::pair_fst> weight;
                weight.weights.push_back(std::make_shared<scrf::linear_score<ilat::pair_fst, scrf::sparse_vec>>(
                    scrf::linear_score<ilat::pair_fst, scrf::sparse_vec>(i_args.param,
                        std::make_shared<scrf::composite_feature<ilat::pair_fst, scrf::sparse_vec>>(graph_feat_func))));
            
                s.graph_data.weight_func = std::make_shared<scrf::composite_weight<ilat::pair_fst>>(weight);
                s.graph_data.feature_func = std::make_shared<scrf::composite_feature<ilat::pair_fst, scrf::sparse_vec>>(
                    graph_feat_func);
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

                l_args.param = scrf::load_sparse_vec(args.at("param"));
                l_args.opt_data = scrf::load_sparse_vec(args.at("opt-data"));
                l_args.step_size = std::stod(args.at("step-size"));
            
                l_args.momentum = -1;
                if (ebt::in(std::string("momentum"), args)) {
                    l_args.momentum = std::stod(args.at("momentum"));
                    assert(0 <= l_args.momentum && l_args.momentum <= 1);
                }
            
                l_args.features = ebt::split(args.at("features"), ",");
            
                l_args.label_id = util::load_label_id(args.at("label"));
            
                l_args.id_label.resize(l_args.label_id.size());
                for (auto& p: l_args.label_id) {
                    l_args.labels.push_back(p.second);
                    l_args.id_label[p.second] = p.first;
                }
            
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
                auto old_weight_func = s.graph_data.weight_func;

                s.graph_data.weight_func = std::make_shared<scrf::mul<ilat::pair_fst>>(scrf::mul<ilat::pair_fst>(
                    std::make_shared<scrf::seg_cost<ilat::pair_fst>>(
                        scrf::make_overlap_cost<ilat::pair_fst>(s.gold_segs, l_args.sils)),
                    -1));

                s.gold_data.fst = scrf::shortest_path(s.graph_data);

                s.graph_data.weight_func = old_weight_func;
            
                using comp_feat = scrf::composite_feature<ilat::pair_fst, scrf::sparse_vec>;

                comp_feat gold_feat_func = make_feat<scrf::sparse_vec, pair_fst_lexicalizer>(
                    s.gold_alloc, l_args.features, s.frames, l_args.args);
            
                s.gold_data.weight_func = std::make_shared<scrf::linear_score<ilat::pair_fst, scrf::sparse_vec>>(
                    scrf::linear_score<ilat::pair_fst, scrf::sparse_vec>(l_args.param,
                    std::make_shared<comp_feat>(gold_feat_func)));
                s.gold_data.feature_func = std::make_shared<comp_feat>(gold_feat_func);
            }

        }

        namespace dense {

            double* pair_fst_lexicalizer::lex(
                scrf::feat_dim_alloc const& alloc, int order,
                scrf::dense_vec& feat, ilat::pair_fst const& fst,
                ilat::pair_fst::edge e) const
            {
                int label_tuple = 0;
                auto const& vertex_attrs = fst.fst2().data->vertex_attrs;
                auto const& id_symbol = *fst.fst1().data->id_symbol;
                auto const& symbol_id = *fst.fst1().data->symbol_id;

                if (order == 0) {
                    if (feat.class_vec.size() < 1) {
                        feat.class_vec.resize(1);
                    }

                    // do nothing
                } else if (order == 1) {
                    label_tuple = fst.output(e) + 1;

                    if (feat.class_vec.size() < 1 + alloc.labels.size()) {
                        feat.class_vec.resize(1 + alloc.labels.size());
                    }
                } else if (order == 2) {
                    if (feat.class_vec.size() < 1 + alloc.labels.size() + alloc.labels.size() * alloc.labels.size()) {
                        feat.class_vec.resize(1 + alloc.labels.size() + alloc.labels.size() * alloc.labels.size());
                    }

                    for (auto& p: vertex_attrs.at(std::get<1>(fst.tail(e)))) {
                        if (p.first == "history") {
                            label_tuple = 1 + alloc.labels.size() + symbol_id.at(p.second) * symbol_id.size();
                            break;
                        }
                    }

                    label_tuple += fst.output(e);
                } else {
                    std::cerr << "order " << order << " not implemented" << std::endl;
                    exit(1);
                }

                la::vector<double>& g = feat.class_vec[label_tuple];
                g.resize(alloc.order_dim[order]);

                return g.data();
            }

        }

    }

}
