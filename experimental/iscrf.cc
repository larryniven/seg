#include "scrf/experimental/iscrf.h"
#include "scrf/experimental/scrf_weight.h"
#include <fstream>

namespace scrf {

    std::vector<int> const& iscrf::vertices() const
    {
        return fst->vertices();
    }

    std::vector<int> const& iscrf::edges() const
    {
        return fst->edges();
    }

    int iscrf::head(int e) const
    {
        return fst->head(e);
    }

    int iscrf::tail(int e) const
    {
        return fst->tail(e);
    }

    std::vector<int> const& iscrf::in_edges(int v) const
    {
        return fst->in_edges(v);
    }

    std::vector<int> const& iscrf::out_edges(int v) const
    {
        return fst->out_edges(v);
    }

    double iscrf::weight(int e) const
    {
        return (*weight_func)(*fst, e);
    }

    int const& iscrf::input(int e) const
    {
        return fst->input(e);
    }

    int const& iscrf::output(int e) const
    {
        return fst->output(e);
    }

    std::vector<int> const& iscrf::initials() const
    {
        return fst->initials();
    }

    std::vector<int> const& iscrf::finals() const
    {
        return fst->finals();
    }

    long iscrf::time(int e) const
    {
        return fst->time(e);
    }

    void iscrf::feature(dense_vec& f, int e) const
    {
        (*feature_func)(f, *fst, e);
    }

    double iscrf::cost(int e) const
    {
        return (*cost_func)(*fst, e);
    }

    std::vector<int> const& iscrf::topo_order() const
    {
        return topo_order_cache;
    }

    std::shared_ptr<iscrf> iscrf_path_maker::operator()(std::vector<int> const& edges,
        iscrf const& f) const
    {
        iscrf result;

        result.fst = ilat::ilat_path_maker()(edges, *f.fst);
        result.topo_order_cache = fst::topo_order(*result.fst);
        result.weight_func = f.weight_func;
        result.feature_func = f.feature_func;
        result.cost_func = f.cost_func;

        return std::make_shared<iscrf>(result);
    }

    std::shared_ptr<ilat::fst> make_graph(int frames,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label,
        int min_seg_len, int max_seg_len)
    {
        ilat::fst_data data;

        data.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(label_id);
        data.id_symbol = std::make_shared<std::vector<std::string>>(id_label);

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

                for (auto& p: label_id) {
                    if (p.second == 0) {
                        continue;
                    }

                    ilat::add_edge(data, data.edges.size(), ilat::edge_data { tail, head, 0, p.second, p.second });
                }
            }
        }

        data.initials.push_back(0);
        data.finals.push_back(frames);

        ilat::fst result;
        result.data = std::make_shared<ilat::fst_data>(std::move(data));

        return std::make_shared<ilat::fst>(result);
    }

    double* ilat_lexicalizer::lex(
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
        std::vector<std::vector<double>> const& frames,
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
                    std::make_shared<segfeat::frame_avg>(
                        segfeat::frame_avg { frames, start_dim, end_dim }),
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
                    std::make_shared<segfeat::frame_samples>(
                        segfeat::frame_samples { 3, start_dim, end_dim }),
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
                    std::make_shared<segfeat::left_boundary>(
                        segfeat::left_boundary { start_dim, end_dim }),
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
                    std::make_shared<segfeat::right_boundary>(
                        segfeat::right_boundary { start_dim, end_dim }),
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
                    std::make_shared<segfeat::length_indicator>(
                        segfeat::length_indicator { max_seg }),
                    frames)));
            } else if (ebt::startswith(k, "bias")) {
                std::vector<std::string> parts = ebt::split(k, "@");
                int order = 0;
                if (parts.size() > 1) {
                    order = std::stoi(parts[1]);
                }

                result.features.push_back(std::make_shared<feat_func>(
                    feat_func(alloc, order,
                    std::make_shared<segfeat::bias>(
                        segfeat::bias {}),
                    frames)));
            } else {
                std::cerr << "unknown feature " << k << std::endl;
                exit(1);
            }
        }

        return result;
    }

    sample::sample(inference_args const& i_args)
        : graph_alloc(i_args.labels)
    {}

    void make_graph(sample& s, inference_args const& i_args)
    {
        s.graph.fst = make_graph(s.frames.size(),
            i_args.label_id, i_args.id_label, i_args.min_seg, i_args.max_seg);
        s.graph.topo_order_cache = ::fst::topo_order(*s.graph.fst);
    }

    void make_lattice(ilat::fst const& lat, sample& s, inference_args const& i_args)
    {
        s.graph.fst = std::make_shared<ilat::fst>(lat);
        s.graph.topo_order_cache = ::fst::topo_order(lat);
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

        l_args.param = load_dense_vec(args.at("param"));
        l_args.opt_data = load_dense_vec(args.at("opt-data"));

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
    {}

    void make_gold(learning_sample& s, learning_args const& l_args)
    {
        s.ground_truth = std::make_shared<iscrf>(iscrf {});
        s.ground_truth->fst = std::make_shared<ilat::fst>(s.ground_truth_fst);

        s.gold = std::make_shared<iscrf>(*s.ground_truth);
    }

    void make_min_cost_gold(learning_sample& s, learning_args const& l_args)
    {
        s.ground_truth = std::make_shared<iscrf>(iscrf {});
        s.ground_truth->fst = std::make_shared<ilat::fst>(s.ground_truth_fst);

        s.graph.weight_func = std::make_shared<neg<ilat::fst>>(neg<ilat::fst>(
            std::make_shared<seg_cost<ilat::fst>>(
                make_overlap_cost<ilat::fst>(s.ground_truth_fst, l_args.sils))));

        s.gold = ::fst::shortest_path<iscrf, iscrf_path_maker>(s.graph);
    }

    void parameterize(iscrf& scrf, feat_dim_alloc& alloc,
        std::vector<std::vector<double>> const& frames,
        inference_args const& i_args)
    {
        using comp_feat = composite_feature<ilat::fst, dense_vec>;

        comp_feat feat_func
            = make_feat(alloc, i_args.features, frames, i_args.args);

        composite_weight<ilat::fst> weight;
        weight.weights.push_back(std::make_shared<linear_score<ilat::fst, dense_vec>>(
            linear_score<ilat::fst, dense_vec>(i_args.param,
            std::make_shared<comp_feat>(feat_func))));
        scrf.weight_func = std::make_shared<composite_weight<ilat::fst>>(weight);
        scrf.feature_func = std::make_shared<comp_feat>(feat_func);
    }

    void parameterize(learning_sample& s, learning_args const& l_args)
    {
        parameterize(s.graph, s.graph_alloc, s.frames, l_args);
        parameterize(*s.gold, s.gold_alloc, s.frames, l_args);
    }

    namespace e2e {

        void iscrf::feature(dense_vec& f, int e) const
        {
            (*feature_func)(f, *fst, e);
        }

        void iscrf::frame_grad(std::vector<std::vector<double>>& f,
            dense_vec const& param, int e) const
        {
            feature_func_with_frame_grad->frame_grad(f, param, *fst, e);
        }

        std::shared_ptr<iscrf> iscrf_path_maker::operator()(std::vector<int> const& edges,
            iscrf const& f) const
        {
            iscrf result;

            result.fst = ilat::ilat_path_maker()(edges, *f.fst);
            result.topo_order_cache = fst::topo_order(*result.fst);
            result.weight_func = f.weight_func;
            result.feature_func = f.feature_func;
            result.cost_func = f.cost_func;

            return std::make_shared<iscrf>(result);
        }

        composite_feature<ilat::fst, dense_vec> make_feat(
            feat_dim_alloc& alloc,
            std::vector<std::string> features,
            std::vector<std::vector<double>> const& frames,
            std::unordered_map<std::string, std::string> const& args)
        {
            composite_feature<ilat::fst, dense_vec> result;

            using feat_func = segment_feature<ilat::fst, dense_vec, ilat_lexicalizer>;

            for (auto& k: features) {
                if (ebt::startswith(k, "frame-samples")) {
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
                        std::make_shared<segfeat::frame_samples>(
                            segfeat::frame_samples { 3, start_dim, end_dim }),
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
                        std::make_shared<segfeat::left_boundary>(
                            segfeat::left_boundary { start_dim, end_dim }),
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
                        std::make_shared<segfeat::right_boundary>(
                            segfeat::right_boundary { start_dim, end_dim }),
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
                        std::make_shared<segfeat::length_indicator>(
                            segfeat::length_indicator { max_seg }),
                        frames)));
                } else if (ebt::startswith(k, "bias")) {
                    std::vector<std::string> parts = ebt::split(k, "@");
                    int order = 0;
                    if (parts.size() > 1) {
                        order = std::stoi(parts[1]);
                    }

                    result.features.push_back(std::make_shared<feat_func>(
                        feat_func(alloc, order,
                        std::make_shared<segfeat::bias>(
                            segfeat::bias {}),
                        frames)));
                }
            }

            return result;
        }

        composite_feature_with_frame_grad<ilat::fst, dense_vec> make_feat_with_frame_grad(
            feat_dim_alloc& alloc,
            std::vector<std::string> features,
            std::vector<std::vector<double>> const& frames,
            std::unordered_map<std::string, std::string> const& args)
        {
            composite_feature_with_frame_grad<ilat::fst, dense_vec> result;

            using feat_func = segment_feature_with_frame_grad<ilat::fst, dense_vec, ilat_lexicalizer>;

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
                        std::make_shared<segfeat::frame_avg>(
                            segfeat::frame_avg { frames, start_dim, end_dim }),
                        frames)));
                }
            }

            return result;
        }

        void parse_inference_args(inference_args& i_args,
            std::unordered_map<std::string, std::string> const& args)
        {
            i_args.args = args;

            i_args.min_seg = 1;
            if (ebt::in(std::string("min-seg"), args)) {
                i_args.min_seg = std::stoi(args.at("min-seg"));
            }

            i_args.max_seg = 20;
            if (ebt::in(std::string("max-seg"), args)) {
                i_args.max_seg = std::stoi(args.at("max-seg"));
            }

            i_args.param = load_dense_vec(args.at("param"));

            std::tie(i_args.nn_param, i_args.pred_param) = load_lstm_param(args.at("nn-param"));

            i_args.features = ebt::split(args.at("features"), ",");

            i_args.label_id = load_label_id(args.at("label"));

            i_args.id_label.resize(i_args.label_id.size());
            for (auto& p: i_args.label_id) {
                i_args.labels.push_back(p.second);
                i_args.id_label[p.second] = p.first;
            }

            if (ebt::in(std::string("subsample-freq"), args)) {
                i_args.subsample_freq = std::stoi(args.at("subsample-freq"));
            }
            if (ebt::in(std::string("subsample-shift"), args)) {
                i_args.subsample_shift = std::stoi(args.at("subsample-shift"));
            }
        }

        sample::sample(inference_args const& i_args)
            : graph_alloc(i_args.labels)
        {}

        void make_graph(sample& s, inference_args const& i_args)
        {
            s.graph.fst = ::scrf::make_graph(s.frames.size(),
                i_args.label_id, i_args.id_label, i_args.min_seg, i_args.max_seg);
            s.graph.topo_order_cache = ::fst::topo_order(*s.graph.fst);
        }

        void make_lattice(ilat::fst const& lat, sample& s, inference_args const& i_args)
        {
            s.graph.fst = std::make_shared<ilat::fst>(lat);
            s.graph.topo_order_cache = ::fst::topo_order(lat);
        }

        void parse_learning_args(learning_args& l_args,
            std::unordered_map<std::string, std::string> const& args)
        {
            parse_inference_args(l_args, args);

            l_args.opt_data = load_dense_vec(args.at("opt-data"));

            std::tie(l_args.nn_opt_data, l_args.pred_opt_data) = load_lstm_param(args.at("nn-opt-data"));

            l_args.step_size = std::stod(args.at("step-size"));

            l_args.momentum = -1;
            if (ebt::in(std::string("momentum"), args)) {
                l_args.momentum = std::stod(args.at("momentum"));
                assert(0 <= l_args.momentum && l_args.momentum <= 1);
            }

            l_args.sils.push_back(l_args.label_id.at("<s>"));
            l_args.sils.push_back(l_args.label_id.at("</s>"));
            l_args.sils.push_back(l_args.label_id.at("sil"));
        }

        std::tuple<lstm::dblstm_feat_param_t, rnn::pred_param_t>
        load_lstm_param(std::string filename)
        {
            std::ifstream ifs { filename };

            lstm::dblstm_feat_param_t nn_param = lstm::load_dblstm_feat_param(ifs);
            rnn::pred_param_t pred_param = rnn::load_pred_param(ifs);

            return std::make_tuple(nn_param, pred_param);
        }

        void save_lstm_param(lstm::dblstm_feat_param_t const& nn_param,
            rnn::pred_param_t const& pred_param,
            std::string filename)
        {
            std::ofstream ofs { filename };

            lstm::save_dblstm_feat_param(nn_param, ofs);
            rnn::save_pred_param(pred_param, ofs);
        }

        learning_sample::learning_sample(learning_args const& args)
            : sample(args), gold_alloc(args.labels)
        {}

        void make_gold(learning_sample& s, learning_args const& l_args)
        {
            s.ground_truth = std::make_shared<iscrf>(iscrf {});
            s.ground_truth->fst = std::make_shared<ilat::fst>(s.ground_truth_fst);

            s.gold = std::make_shared<iscrf>(*s.ground_truth);
        }

        void make_min_cost_gold(learning_sample& s, learning_args const& l_args)
        {
            s.ground_truth = std::make_shared<iscrf>(iscrf {});
            s.ground_truth->fst = std::make_shared<ilat::fst>(s.ground_truth_fst);

            s.graph.weight_func = std::make_shared<neg<ilat::fst>>(neg<ilat::fst>(
                std::make_shared<seg_cost<ilat::fst>>(
                    make_overlap_cost<ilat::fst>(s.ground_truth_fst, l_args.sils))));

            s.gold = ::fst::shortest_path<iscrf, iscrf_path_maker>(s.graph);
        }

        void parameterize(iscrf& scrf, feat_dim_alloc& alloc,
            std::vector<std::vector<double>> const& frames,
            inference_args const& i_args)
        {
            using comp_feat = composite_feature<ilat::fst, dense_vec>;
            using comp_feat_with_frame_grad = composite_feature_with_frame_grad<ilat::fst, dense_vec>;

            comp_feat feat_func
                = ::scrf::e2e::make_feat(alloc, i_args.features, frames, i_args.args);
            comp_feat_with_frame_grad feat_func_with_frame_grad
                = make_feat_with_frame_grad(alloc, i_args.features, frames, i_args.args);
            for (auto& f: feat_func_with_frame_grad.features) {
                feat_func.features.push_back(f);
            }

            composite_weight<ilat::fst> weight;
            weight.weights.push_back(std::make_shared<linear_score<ilat::fst, dense_vec>>(
                linear_score<ilat::fst, dense_vec>(i_args.param,
                std::make_shared<comp_feat>(feat_func))));
            scrf.weight_func = std::make_shared<composite_weight<ilat::fst>>(weight);
            scrf.feature_func = std::make_shared<comp_feat>(feat_func);
            scrf.feature_func_with_frame_grad = std::make_shared<comp_feat_with_frame_grad>(
                feat_func_with_frame_grad);
        }

        void parameterize(learning_sample& s, learning_args const& l_args)
        {
            parameterize(s.graph, s.graph_alloc, s.frames, l_args);
            parameterize(*s.gold, s.gold_alloc, s.frames, l_args);
        }

    }

}
