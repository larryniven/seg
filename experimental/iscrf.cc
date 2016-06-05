#include "scrf/experimental/iscrf.h"
#include "scrf/experimental/scrf_weight.h"
#include <fstream>

namespace iscrf {

     double weight(iscrf_data const& data, int e)
     {
          return (*data.weight_func)(*data.fst, e);
     }
 
     void feature(iscrf_data const& data, scrf::dense_vec& f, int e)
     {
          (*data.feature_func)(f, *data.fst, e);
     }
 
     double cost(iscrf_data const& data, int e)
     {
          return (*data.cost_func)(*data.fst, e);
     }
 
    std::shared_ptr<ilat::fst> make_graph(int frames,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label,
        int min_seg_len, int max_seg_len, int stride)
    {
        assert(stride >= 1);
        assert(min_seg_len >= 1);
        assert(max_seg_len >= min_seg_len);

        ilat::fst_data data;

        data.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(label_id);
        data.id_symbol = std::make_shared<std::vector<std::string>>(id_label);

        int i = 0;
        int v = -1;
        for (i = 0; i < frames + 1; i += stride) {
            ++v;
            ilat::add_vertex(data, v, ilat::vertex_data { i });
        }

        if (frames % stride != 0) {
            ++v;
            ilat::add_vertex(data, v, ilat::vertex_data { frames });
        }

        data.initials.push_back(0);
        data.finals.push_back(v);

        for (int u = 0; u < data.vertices.size(); ++u) {
            for (int v = u + 1; v < data.vertices.size(); ++v) {
                int duration = data.vertices[v].time - data.vertices[u].time;

                if (duration < min_seg_len) {
                    continue;
                }

                if (duration > max_seg_len) {
                    break;
                }

                for (auto& p: label_id) {
                    if (p.second == 0) {
                        continue;
                    }

                    ilat::add_edge(data, data.edges.size(),
                        ilat::edge_data { u, v, 0, p.second, p.second });
                }
            }
        }

        ilat::fst result;
        result.data = std::make_shared<ilat::fst_data>(std::move(data));

        return std::make_shared<ilat::fst>(result);
    }

    double* ilat_lexicalizer::lex(
        scrf::feat_dim_alloc const& alloc, int order,
        scrf::dense_vec& feat, ilat::fst const& fst, int e) const
    {
        int label_tuple = 0;

        if (order == 0) {
            if (feat.class_vec.size() < 1) {
                feat.class_vec.resize(1);
            }
        } else if (order == 1) {
            label_tuple = fst.output(e) + 1;
            if (feat.class_vec.size() < alloc.labels.size() + 1) {
                feat.class_vec.resize(alloc.labels.size() + 1);
            }
        } else {
            std::cerr << "order " << order << " not implemented" << std::endl;
            exit(1);
        }

        la::vector<double>& g = feat.class_vec[label_tuple];
        g.resize(alloc.order_dim[order]);

        return g.data();
    }

    scrf::composite_feature<ilat::fst, scrf::dense_vec> make_feat(
        scrf::feat_dim_alloc& alloc,
        std::vector<std::string> features,
        std::vector<std::vector<double>> const& frames,
        std::unordered_map<std::string, std::string> const& args)
    {
        scrf::composite_feature<ilat::fst, scrf::dense_vec> result;

        using feat_func = scrf::segment_feature<ilat::fst, scrf::dense_vec, ilat_lexicalizer>;
        using feat_func_with_frame_grad = scrf::segment_feature_with_frame_grad<
            ilat::fst, scrf::dense_vec, ilat_lexicalizer>;

        for (auto& k: features) {
            if (ebt::startswith(k, "frame-avg")) {
                std::vector<std::string> parts = ebt::split(k, "@");
                int order = 0;
                if (parts.size() > 1) {
                    order = std::stoi(parts[1]);
                }

                int start_dim = -1;
                int end_dim = -1;
                std::tie(start_dim, end_dim) = scrf::get_dim(parts[0]);

                result.features.push_back(std::make_shared<feat_func_with_frame_grad>(
                    feat_func_with_frame_grad(alloc, order,
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
                std::tie(start_dim, end_dim) = scrf::get_dim(parts[0]);

                result.features.push_back(std::make_shared<feat_func_with_frame_grad>(
                    feat_func_with_frame_grad(alloc, order,
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
                std::tie(start_dim, end_dim) = scrf::get_dim(parts[0]);

                result.features.push_back(std::make_shared<feat_func_with_frame_grad>(
                    feat_func_with_frame_grad(alloc, order,
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
                std::tie(start_dim, end_dim) = scrf::get_dim(parts[0]);

                result.features.push_back(std::make_shared<feat_func_with_frame_grad>(
                    feat_func_with_frame_grad(alloc, order,
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

        i_args.stride = 1;
        if (ebt::in(std::string("stride"), args)) {
            i_args.stride = std::stoi(args.at("stride"));
        }

        i_args.param = scrf::load_dense_vec(args.at("param"));

        i_args.features = ebt::split(args.at("features"), ",");

        i_args.label_id = scrf::load_label_id(args.at("label"));

        i_args.id_label.resize(i_args.label_id.size());
        for (auto& p: i_args.label_id) {
            i_args.labels.push_back(p.second);
            i_args.id_label[p.second] = p.first;
        }
    }

    sample::sample(inference_args const& i_args)
        : graph_alloc(i_args.labels)
    {
        graph_data.features = std::make_shared<std::vector<std::string>>(i_args.features);
        graph_data.param = std::make_shared<scrf::dense_vec>(i_args.param);
    }

    void make_graph(sample& s, inference_args const& i_args)
    {
        s.graph_data.fst = make_graph(s.frames.size(),
            i_args.label_id, i_args.id_label, i_args.min_seg, i_args.max_seg, i_args.stride);
        s.graph_data.topo_order = std::make_shared<std::vector<int>>(::fst::topo_order(*s.graph_data.fst));
    }

    void make_lattice(ilat::fst const& lat, sample& s, inference_args const& i_args)
    {
        s.graph_data.fst = std::make_shared<ilat::fst>(lat);
        s.graph_data.topo_order = std::make_shared<std::vector<int>>(::fst::topo_order(lat));
    }

    std::vector<segcost::segment<int>> load_segments(std::istream& is,
        std::unordered_map<std::string, int> const& label_id)
    {
        std::string line;

        std::vector<segcost::segment<int>> result;

        // get rid of the name
        std::getline(is, line);

        while (std::getline(is, line) && line != ".") {
            auto parts = ebt::split(line);

            result.push_back(segcost::segment<int> {
                 .start_time = std::stoi(parts[0]),
                 .end_time = std::stoi(parts[1]),
                 .label = label_id.at(parts[2])
            });
        }

        return result;
    }

    std::vector<segcost::segment<std::string>> load_segments(std::istream& is)
    {
        std::string line;

        std::vector<segcost::segment<std::string>> result;

        // get rid of the name
        std::getline(is, line);

        while (std::getline(is, line) && line != ".") {
            auto parts = ebt::split(line);

            result.push_back(segcost::segment<std::string> {
                 .start_time = std::stoi(parts[0]),
                 .end_time = std::stoi(parts[1]),
                 .label = parts[2]
            });
        }

        return result;
    }

    void parse_learning_args(learning_args& l_args,
        std::unordered_map<std::string, std::string> const& args)
    {
        parse_inference_args(l_args, args);

        if (ebt::in(std::string("opt-data"), args)) {
            l_args.opt_data = scrf::load_dense_vec(args.at("opt-data"));
        }

        l_args.step_size = 0;
        if (ebt::in(std::string("step-size"), args)) {
            l_args.step_size = std::stod(args.at("step-size"));
        }

        l_args.momentum = -1;
        if (ebt::in(std::string("momentum"), args)) {
            l_args.momentum = std::stod(args.at("momentum"));
            assert(0 <= l_args.momentum && l_args.momentum <= 1);
        }

        l_args.decay = -1;
        if (ebt::in(std::string("decay"), args)) {
            l_args.decay = std::stod(args.at("decay"));
            assert(0 <= l_args.decay && l_args.decay <= 1);
        }

        l_args.cost_scale = 1;
        if (ebt::in(std::string("cost-scale"), args)) {
            l_args.cost_scale = std::stod(args.at("cost-scale"));
            assert(l_args.cost_scale >= 0);
        }

        if (ebt::in(std::string("sil"), l_args.label_id)) {
            l_args.sils.push_back(l_args.label_id.at("sil"));
        }
    }

    learning_sample::learning_sample(learning_args const& l_args)
        : sample(l_args), gold_alloc(l_args.labels)
    {
        gold_data.features = std::make_shared<std::vector<std::string>>(l_args.features);
        gold_data.param = std::make_shared<scrf::dense_vec>(l_args.param);
    }

    void make_min_cost_gold(learning_sample& s, learning_args const& l_args)
    {
        auto old_weight_func = s.graph_data.weight_func;

        s.graph_data.weight_func = std::make_shared<scrf::mul<ilat::fst>>(scrf::mul<ilat::fst>(
            std::make_shared<scrf::seg_cost<ilat::fst>>(
                scrf::make_overlap_cost<ilat::fst>(s.gold_segs, l_args.sils)),
            -1));

        s.gold_data.fst = scrf::shortest_path(s.graph_data);

        s.graph_data.weight_func = old_weight_func;
    }

    void parameterize(iscrf_data& data, scrf::feat_dim_alloc& alloc,
        std::vector<std::vector<double>> const& frames,
        inference_args const& i_args)
    {
        using comp_feat = scrf::composite_feature<ilat::fst, scrf::dense_vec>;

        comp_feat feat_func
            = make_feat(alloc, i_args.features, frames, i_args.args);

        scrf::composite_weight<ilat::fst> weight;
        weight.weights.push_back(std::make_shared<scrf::linear_score<ilat::fst, scrf::dense_vec>>(
            scrf::linear_score<ilat::fst, scrf::dense_vec>(i_args.param,
            std::make_shared<comp_feat>(feat_func))));

        data.weight_func = std::make_shared<scrf::composite_weight<ilat::fst>>(weight);
        data.feature_func = std::make_shared<comp_feat>(feat_func);
    }

    void parameterize(learning_sample& s, learning_args const& l_args)
    {
        parameterize(s.graph_data, s.graph_alloc, s.frames, l_args);

        if (!ebt::in(std::string("use-gold-segs"), l_args.args)) {
            std::vector<segcost::segment<int>> min_cost_segs;

            for (auto& e: s.gold_data.fst->edges()) {
                min_cost_segs.push_back(segcost::segment<int> {
                    s.gold_data.fst->time(s.gold_data.fst->tail(e)),
                    s.gold_data.fst->time(s.gold_data.fst->head(e)),
                    s.gold_data.fst->output(e)
                });
            }

            s.gold_segs = min_cost_segs;
        }

        s.graph_data.cost_func = std::make_shared<scrf::mul<ilat::fst>>(scrf::mul<ilat::fst>(
            std::make_shared<scrf::seg_cost<ilat::fst>>(
                scrf::make_overlap_cost<ilat::fst>(s.gold_segs, l_args.sils)),
            l_args.cost_scale));

        parameterize(s.gold_data, s.gold_alloc, s.frames, l_args);

        s.gold_data.cost_func = std::make_shared<scrf::mul<ilat::fst>>(scrf::mul<ilat::fst>(
            std::make_shared<scrf::seg_cost<ilat::fst>>(
                scrf::make_overlap_cost<ilat::fst>(s.gold_segs, l_args.sils)),
            l_args.cost_scale));
    }

}
