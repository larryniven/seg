#include "scrf/pair_scrf.h"
#include "scrf/scrf_weight.h"

namespace scrf {

    namespace experimental {

        namespace second_order {

            std::vector<pair_scrf::vertex> const& pair_scrf::vertices() const
            {
                return fst->vertices();
            }

            std::vector<pair_scrf::edge> const& pair_scrf::edges() const
            {
                return fst->edges();
            }

            pair_scrf::vertex pair_scrf::head(pair_scrf::edge e) const
            {
                return fst->head(e);
            }

            pair_scrf::vertex pair_scrf::tail(pair_scrf::edge e) const
            {
                return fst->tail(e);
            }

            std::vector<pair_scrf::edge> const& pair_scrf::in_edges(pair_scrf::vertex v) const
            {
                return fst->in_edges(v);
            }

            std::vector<pair_scrf::edge> const& pair_scrf::out_edges(pair_scrf::vertex v) const
            {
                return fst->out_edges(v);
            }

            double pair_scrf::weight(pair_scrf::edge e) const
            {
                return (*weight_func)(*fst, e);
            }

            int const& pair_scrf::input(pair_scrf::edge e) const
            {
                return fst->input(e);
            }

            int const& pair_scrf::output(pair_scrf::edge e) const
            {
                return fst->output(e);
            }

            std::vector<pair_scrf::vertex> const& pair_scrf::initials() const
            {
                return fst->initials();
            }

            std::vector<pair_scrf::vertex> const& pair_scrf::finals() const
            {
                return fst->finals();
            }

            long pair_scrf::time(pair_scrf::edge e) const
            {
                return fst->time(e);
            }

            void pair_scrf::feature(sparse_vec& f, pair_scrf::edge e) const
            {
                (*feature_func)(f, *fst, e);
            }

            double pair_scrf::cost(pair_scrf::edge e) const
            {
                return (*cost_func)(*fst, e);
            }

            std::vector<pair_scrf::vertex> const& pair_scrf::topo_order() const
            {
                return topo_order_cache;
            }

            std::shared_ptr<pair_scrf> pair_scrf_path_maker::operator()(std::vector<pair_scrf::edge> const& edges,
                pair_scrf const& f) const
            {
                pair_scrf result;

                result.fst = ilat::pair_fst_path_maker()(edges, *f.fst);
                result.topo_order_cache = fst::topo_order(*result.fst);
                result.weight_func = f.weight_func;
                result.feature_func = f.feature_func;
                result.cost_func = f.cost_func;

                return std::make_shared<pair_scrf>(result);
            }

            double* pair_fst_lexicalizer::operator()(
                feat_dim_alloc const& alloc, int order,
                sparse_vec& feat, ilat::pair_fst const& fst,
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

            std::pair<int, int> get_dim(std::string feat)
            {
                std::vector<std::string> parts = ebt::split(feat, ":");
                int start_dim = -1;
                int end_dim = -1;
                if (parts.size() == 2) {
                    std::vector<std::string> indices = ebt::split(parts.back(), "-");
                    start_dim = std::stoi(indices.at(0));
                    end_dim = std::stoi(indices.at(1));
                }

                return std::make_pair(start_dim, end_dim);
            }

            composite_feature<ilat::pair_fst, sparse_vec> make_feat(
                feat_dim_alloc& alloc,
                std::vector<std::string> features,
                std::vector<std::vector<real>> const& frames,
                std::unordered_map<std::string, std::string> const& args)
            {
                composite_feature<ilat::pair_fst, sparse_vec> result;

                using feat_func = segment_feature<ilat::pair_fst, sparse_vec, pair_fst_lexicalizer>;

                for (auto& k: features) {
                    if (ebt::startswith(k, "frame-avg")) {
                        std::vector<std::string> parts = ebt::split(k, "@");
                        int order = 0;
                        if (parts.size() > 1) {
                            order = std::stoi(parts[1]);
                        }

                        int start_dim = -1;
                        int end_dim = -1;
                        std::tie(start_dim, end_dim) = get_dim(parts[0]);

                        result.features.push_back(std::make_shared<feat_func>(
                            feat_func(alloc, order,
                            std::make_shared<segfeat::experimental::frame_avg>(
                                segfeat::experimental::frame_avg { frames, start_dim, end_dim }),
                            frames)));
                    } else if (ebt::startswith(k, "frame-samples")) {
                        std::vector<std::string> parts = ebt::split(k, "@");
                        int order = 0;
                        if (parts.size() > 1) {
                            order = std::stoi(parts[1]);
                        }

                        int start_dim = -1;
                        int end_dim = -1;
                        std::tie(start_dim, end_dim) = get_dim(parts[0]);

                        result.features.push_back(std::make_shared<feat_func>(
                            feat_func(alloc, order,
                            std::make_shared<segfeat::experimental::frame_samples>(
                                segfeat::experimental::frame_samples { 3, start_dim, end_dim }),
                            frames)));
                    } else if (ebt::startswith(k, "left-boundary")) {
                        std::vector<std::string> parts = ebt::split(k, "@");
                        int order = 0;
                        if (parts.size() > 1) {
                            order = std::stoi(parts[1]);
                        }

                        int start_dim = -1;
                        int end_dim = -1;
                        std::tie(start_dim, end_dim) = get_dim(parts[0]);

                        result.features.push_back(std::make_shared<feat_func>(
                            feat_func(alloc, order,
                            std::make_shared<segfeat::experimental::left_boundary>(
                                segfeat::experimental::left_boundary { start_dim, end_dim }),
                            frames)));
                    } else if (ebt::startswith(k, "right-boundary")) {
                        std::vector<std::string> parts = ebt::split(k, "@");
                        int order = 0;
                        if (parts.size() > 1) {
                            order = std::stoi(parts[1]);
                        }

                        int start_dim = -1;
                        int end_dim = -1;
                        std::tie(start_dim, end_dim) = get_dim(parts[0]);

                        result.features.push_back(std::make_shared<feat_func>(
                            feat_func(alloc, order,
                            std::make_shared<segfeat::experimental::right_boundary>(
                                segfeat::experimental::right_boundary { start_dim, end_dim }),
                            frames)));
                    } else if (ebt::startswith(k, "length-indicator")) {
                        std::vector<std::string> parts = ebt::split(k, "@");
                        int order = 0;
                        if (parts.size() > 1) {
                            order = std::stoi(parts[1]);
                        }

                        if (!ebt::in(std::string("max-seg"), args)) {
                            std::cerr << "--max-seg is required" << std::endl;
                            exit(1);
                        }

                        int max_seg = std::stoi(args.at("max-seg"));

                        result.features.push_back(std::make_shared<feat_func>(
                            feat_func(alloc, order,
                            std::make_shared<segfeat::experimental::length_indicator>(
                                segfeat::experimental::length_indicator { max_seg }),
                            frames)));
                    } else if (ebt::startswith(k, "bias")) {
                        std::vector<std::string> parts = ebt::split(k, "@");
                        int order = 0;
                        if (parts.size() > 1) {
                            order = std::stoi(parts[1]);
                        }

                        result.features.push_back(std::make_shared<feat_func>(
                            feat_func(alloc, order,
                            std::make_shared<segfeat::experimental::bias>(
                                segfeat::experimental::bias {}),
                            frames)));
                    } else {
                        std::cerr << "unknown feature " << k << std::endl;
                        exit(1);
                    }
                }

                return result;
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
                    = make_feat(s.graph_alloc, i_args.features, s.frames, i_args.args);
            
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
            
            void make_gold(learning_sample& s, learning_args const& l_args)
            {
                ilat::fst ground_truth_with_loops;
                ground_truth_with_loops.data = std::make_shared<ilat::fst_data>(*s.ground_truth_fst.data);
                add_eps_loops(ground_truth_with_loops);

                ilat::lazy_pair comp_lat { ground_truth_with_loops, l_args.lm };

                pair_scrf gt_lm;
                gt_lm.fst = std::make_shared<ilat::lazy_pair>(comp_lat);

                auto lat_order = fst::topo_order(s.ground_truth_fst);
                for (auto& v: lat_order) {
                    for (int i = l_args.lm.vertices().size() - 1; i >= 0; --i) {
                        gt_lm.topo_order_cache.push_back(std::make_tuple(v, i));
                    }
                }

                gt_lm.weight_func = std::make_shared<neg<ilat::pair_fst>>(
                    neg<ilat::pair_fst>(std::make_shared<backoff_cost>(backoff_cost{})));

                s.ground_truth = fst::experimental::shortest_path<pair_scrf, pair_scrf_path_maker>(gt_lm);

                std::cout << "ground truth path: ";
                for (auto& e: s.ground_truth->edges()) {
                    std::cout << l_args.id_label.at(s.ground_truth->output(e)) << " ";
                }
                std::cout << std::endl;

                s.gold = fst::experimental::shortest_path<pair_scrf, pair_scrf_path_maker>(*s.ground_truth);

                using comp_feat = composite_feature<ilat::pair_fst, sparse_vec>;

                comp_feat gold_feat_func = make_feat(s.gold_alloc, l_args.features, s.frames, l_args.args);
            
                s.gold->weight_func = std::make_shared<linear_score<ilat::pair_fst, sparse_vec>>(
                    linear_score<ilat::pair_fst, sparse_vec>(l_args.param,
                    std::make_shared<comp_feat>(gold_feat_func)));
                s.gold->feature_func = std::make_shared<comp_feat>(gold_feat_func);
            }

            void make_min_cost_gold(learning_sample& s, learning_args const& l_args)
            {
                ilat::fst ground_truth_with_loops;
                ground_truth_with_loops.data = std::make_shared<ilat::fst_data>(*s.ground_truth_fst.data);
                add_eps_loops(ground_truth_with_loops);

                ilat::lazy_pair comp_lat { ground_truth_with_loops, l_args.lm };

                pair_scrf gt_lm;
                gt_lm.fst = std::make_shared<ilat::lazy_pair>(comp_lat);

                auto lat_order = fst::topo_order(s.ground_truth_fst);
                for (auto& v: lat_order) {
                    for (int i = l_args.lm.vertices().size() - 1; i >= 0; --i) {
                        gt_lm.topo_order_cache.push_back(std::make_tuple(v, i));
                    }
                }

                gt_lm.weight_func = std::make_shared<neg<ilat::pair_fst>>(
                    neg<ilat::pair_fst>(std::make_shared<backoff_cost>(backoff_cost{})));

                s.ground_truth = fst::experimental::shortest_path<pair_scrf, pair_scrf_path_maker>(gt_lm);

                std::cout << "ground truth path: ";
                for (auto& e: s.ground_truth->edges()) {
                    std::cout << l_args.id_label.at(s.ground_truth->output(e)) << " ";
                }
                std::cout << std::endl;

                auto old_weight_func = s.graph.weight_func;

                s.graph.weight_func = std::make_shared<neg<ilat::pair_fst>>(neg<ilat::pair_fst>(
                    std::make_shared<seg_cost<ilat::pair_fst>>(
                        make_overlap_cost<ilat::pair_fst>(*s.ground_truth->fst, l_args.sils))));

                s.gold = fst::experimental::shortest_path<pair_scrf, pair_scrf_path_maker>(s.graph);

                std::cout << "gold path: ";
                double gold_path_weight = 0;
                for (auto& e: s.gold->edges()) {
                    std::cout << l_args.id_label.at(s.gold->output(e)) << " ";
                    gold_path_weight += s.gold->weight(e);
                }
                std::cout << std::endl;
                std::cout << "gold cost: " << gold_path_weight << std::endl;

                s.graph.weight_func = old_weight_func;
            
                using comp_feat = composite_feature<ilat::pair_fst, sparse_vec>;

                comp_feat gold_feat_func = make_feat(s.gold_alloc, l_args.features, s.frames, l_args.args);
            
                s.gold->weight_func = std::make_shared<linear_score<ilat::pair_fst, sparse_vec>>(
                    linear_score<ilat::pair_fst, sparse_vec>(l_args.param,
                    std::make_shared<comp_feat>(gold_feat_func)));
                s.gold->feature_func = std::make_shared<comp_feat>(gold_feat_func);
            }

        }

    }

}
