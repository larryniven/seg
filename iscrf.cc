#include "scrf/iscrf.h"
#include "scrf/scrf_weight.h"

namespace scrf {

    namespace experimental {

        std::vector<int> const& iscrf::vertices() const
        {
            return fst.vertices();
        }

        std::vector<int> const& iscrf::edges() const
        {
            return fst.edges();
        }

        int iscrf::head(int e) const
        {
            return fst.head(e);
        }

        int iscrf::tail(int e) const
        {
            return fst.tail(e);
        }

        std::vector<int> const& iscrf::in_edges(int v) const
        {
            return fst.in_edges(v);
        }

        std::vector<int> const& iscrf::out_edges(int v) const
        {
            return fst.out_edges(v);
        }

        double iscrf::weight(int e) const
        {
            return (*weight_func)(fst, e);
        }

        int const& iscrf::input(int e) const
        {
            return fst.input(e);
        }

        int const& iscrf::output(int e) const
        {
            return fst.output(e);
        }

        std::vector<int> const& iscrf::initials() const
        {
            return fst.initials();
        }

        std::vector<int> const& iscrf::finals() const
        {
            return fst.finals();
        }

        long iscrf::time(int e) const
        {
            return fst.time(e);
        }

        void iscrf::feature(dense_vec& f, int e) const
        {
            (*feature_func)(f, fst, e);
        }

        double iscrf::cost(int e) const
        {
            return (*cost_func)(fst, e);
        }

        std::vector<int> const& iscrf::topo_order() const
        {
            return topo_order_cache;
        }

        iscrf iscrf_path_maker::operator()(std::vector<int> const& edges, iscrf const& f) const
        {
            iscrf result;

            result.fst = ilat::ilat_path_maker()(edges, f.fst);
            result.topo_order_cache = fst::topo_order(result.fst);
            result.weight_func = f.weight_func;
            result.feature_func = f.feature_func;
            result.cost_func = f.cost_func;

            return result;
        }

        iscrf iscrf_graph_maker::operator()(int frames,
            std::vector<int> const& labels,
            int min_seg_len, int max_seg_len) const
        {
            ilat::fst_data data;

            for (int i = 0; i < frames + 1; ++i) {
                ilat::add_vertex(data, i, ilat::vertex_data { i });
            }

            assert(min_seg_len >= 1);

            for (int i = 0; i < frames + 1; ++i) {
                for (int j = min_seg_len; j <= max_seg_len; ++j) {
                    int tail = i;
                    int head = i + j;

                    if (head > frames) {
                        continue;
                    }

                    for (auto& ell: labels) {
                        ilat::add_edge(data, data.edges.size(), ilat::edge_data { tail, head, 0, ell });
                    }
                }
            }

            data.initials.push_back(0);
            data.finals.push_back(frames);

            iscrf result;

            result.fst.data = std::make_shared<ilat::fst_data>(std::move(data));
            result.topo_order_cache = fst::topo_order(result.fst);

            return result;
        }

        double* ilat_lexicalizer::operator()(
            feat_dim_alloc const& alloc, int order,
            dense_vec& feat, ilat::fst const& fst, int e) const
        {
            int label_tuple = 0;

            if (order == 0) {
                feat.class_vec.resize(1);
            } else if (order == 1) {
                label_tuple = fst.output(e) + 1;
                feat.class_vec.resize(alloc.labels.size() + 1);
            } else {
                std::cerr << "order " << order << " not implemented" << std::endl;
                exit(1);
            }

            la::vector<double>& g = feat.class_vec[label_tuple];
            g.resize(alloc.order_dim[order]);

            return g.data();
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

        composite_feature<ilat::fst, dense_vec> make_feat(
            feat_dim_alloc& alloc,
            std::vector<std::string> features,
            std::vector<std::vector<real>> const& frames,
            std::unordered_map<std::string, std::string> const& args)
        {
            composite_feature<ilat::fst, dense_vec> result;

            using feat_func = segment_feature<ilat::fst, dense_vec, ilat_lexicalizer>;

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
        
        void make_graph(sample& s, inference_args const& i_args)
        {
            s.graph = iscrf_graph_maker()(s.frames.size(),
                i_args.labels, i_args.min_seg, i_args.max_seg);
        
            composite_feature<ilat::fst, dense_vec> graph_feat_func
                = make_feat(s.graph_alloc, i_args.features, s.frames, i_args.args);
        
            composite_weight<ilat::fst> weight;
            weight.weights.push_back(std::make_shared<linear_score<ilat::fst, dense_vec>>(
                linear_score<ilat::fst, dense_vec>(i_args.param,
                    std::make_shared<composite_feature<ilat::fst, dense_vec>>(graph_feat_func))));
        
            s.graph.weight_func = std::make_shared<composite_weight<ilat::fst>>(weight);
            s.graph.feature_func = std::make_shared<composite_feature<ilat::fst, dense_vec>>(graph_feat_func);
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
        
            l_args.param = load_vec(args.at("param"));
            l_args.opt_data = load_vec(args.at("opt-data"));
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
        
            return l_args;
        }
        
        learning_sample::learning_sample(learning_args const& args)
            : sample(args), gold_alloc(args.labels)
        {
        }
        
        void make_gold(learning_sample& s, learning_args const& l_args)
        {
            s.ground_truth.fst = s.ground_truth_fst;
        
            s.gold = s.ground_truth;
        
            composite_feature<ilat::fst, dense_vec> gold_feat_func
                = make_feat(s.gold_alloc, l_args.features, s.frames, l_args.args);
        
            s.gold.weight_func = std::make_shared<linear_score<ilat::fst, dense_vec>>(
                linear_score<ilat::fst, dense_vec>(l_args.param,
                    std::make_shared<composite_feature<ilat::fst, dense_vec>>(gold_feat_func)));
            s.gold.feature_func = std::make_shared<composite_feature<ilat::fst, dense_vec>>(gold_feat_func);
        }
        
        void make_min_cost_gold(learning_sample& s, learning_args const& l_args)
        {
            s.ground_truth.fst = s.ground_truth_fst;
        
            iscrf graph = iscrf_graph_maker()(s.frames.size(), l_args.labels, l_args.min_seg, l_args.max_seg);
            graph.weight_func = std::make_shared<neg<ilat::fst>>(neg<ilat::fst>(
                std::make_shared<seg_cost<ilat::fst>>(
                    make_overlap_cost<ilat::fst>(s.ground_truth_fst, l_args.sils))));

            s.gold = fst::experimental::shortest_path<iscrf, iscrf_path_maker>(graph);
        
            composite_feature<ilat::fst, dense_vec> gold_feat_func
                = make_feat(s.gold_alloc, l_args.features, s.frames, l_args.args);
        
            s.gold.weight_func = std::make_shared<linear_score<ilat::fst, dense_vec>>(
                linear_score<ilat::fst, dense_vec>(l_args.param,
                std::make_shared<composite_feature<ilat::fst, dense_vec>>(gold_feat_func)));
            s.gold.feature_func = std::make_shared<composite_feature<ilat::fst, dense_vec>>(gold_feat_func);
        }

    }

}
