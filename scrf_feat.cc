#include "scrf/scrf_feat.h"
#include <cassert>
#include "scrf/scrf_feat_util.h"
#include <fstream>

namespace scrf {

    composite_feature::composite_feature()
    {}

    void composite_feature::operator()(
        feat_t& feat,
        fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        for (auto& f: features) {
            (*f)(feat, fst, e);
        }
    }
        
    std::vector<double>& lexicalize(int order, feat_t& feat,
        fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e)
    {
        std::string label_tuple;

        if (order == 0) {
            // do nothing
        } else if (order == 1) {
            label_tuple = fst.output(e);
        } else if (order == 2) {
            auto const& lm = *fst.fst2;
            label_tuple = lm.data->history.at(std::get<1>(fst.tail(e))) + "_" + fst.output(e);
        } else {
            std::cerr << "order " << order << " not implemented" << std::endl;
            exit(1);
        }

        auto& g = feat.class_vec[label_tuple];
        g.reserve(1000);

        return g;
    }

    segment_feature::segment_feature(
        int order,
        std::shared_ptr<segfeat::feature> feat_func,
        std::vector<std::vector<real>> const& frames)
        : order(order), feat_func(feat_func), frames(frames)
    {}

    void segment_feature::operator()(
        feat_t& feat,
        fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        std::vector<double>& g = lexicalize(order, feat, fst, e);

        lattice::fst const& lat = *fst.fst1;
        int tail_time = lat.data->vertices.at(std::get<0>(fst.tail(e))).time;
        int head_time = lat.data->vertices.at(std::get<0>(fst.head(e))).time;

        (*feat_func)(g, frames, tail_time, head_time);
    }

    namespace feature {

        void lm_score::operator()(
            feat_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            feat.class_vec[""].push_back(fst.fst2->weight(std::get<1>(e)));
        }

        void lattice_score::operator()(
            feat_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            feat.class_vec[""].push_back(fst.fst1->weight(std::get<0>(e)));
        }

        external_feature::external_feature(int order, std::vector<int> dims)
            : order(order), dims(dims)
        {}

        void external_feature::operator()(
            feat_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            auto& g = lexicalize(order, feat, fst, e);

            lattice::fst& lat = *fst.fst1;

            std::vector<double> const& f = lat.data->feats[std::get<0>(e)];

            if (fst.input(e) == "<eps>") {
                for (auto i: dims) {
                    g.push_back(0);
                }
            } else {
                for (auto i: dims) {
                    g.push_back(f[i]);
                }
            }
        }

        frame_feature::frame_feature(std::vector<std::vector<double>> const& frames,
            std::unordered_map<std::string, std::string> const& args)
            : frames(frames)
        {
            if (!ebt::in(std::string("label-dim"), args)) {
                std::cerr << "--label-dim missing" << std::endl;
                exit(1);
            }

            label_dim = load_label_dim(args.at("label-dim"));
        }

        std::unordered_map<std::string, std::vector<int>>
        frame_feature::load_label_dim(std::string filename)
        {
            std::unordered_map<std::string, std::vector<int>> result;
            std::string line;
            std::ifstream ifs { filename };
        
            while (std::getline(ifs, line)) {
                auto parts = ebt::split(line);
                std::vector<int> dims;
                for (int i = 1; i < parts.size(); ++i) {
                    dims.push_back(std::stoi(parts[i]));
                }
                result[parts[0]] = dims;
            }
        
            return result;
        }

        void frame_feature::operator()(
            scrf::feat_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            auto& lat = *fst.fst1;
            int tail_time = std::min<int>(frames.size() - 1, lat.data->vertices.at(std::get<0>(fst.tail(e))).time);
            int head_time = std::min<int>(frames.size(), lat.data->vertices.at(std::get<0>(fst.head(e))).time);

            double result = 0;

            for (int i = tail_time; i < head_time; ++i) {
                for (int dim: label_dim.at(fst.output(e))) {
                    result += frames[i][dim];
                }
            }

            feat.class_vec[""].push_back(result);
        }

        quad_length::quad_length(int order,
            std::unordered_map<std::string, std::string> const& args)
            : order(order)
        {
            if (!ebt::in(std::string("length-stat"), args)) {
                std::cerr << "--length-stat" << std::endl;
                exit(1);
            }

            std::tie(mean, var) = load_length_stat(args.at("length-stat"));
        }

        std::tuple<std::unordered_map<std::string, double>,
            std::unordered_map<std::string, double>>
        quad_length::load_length_stat(std::string filename) const
        {
            std::ifstream ifs { filename };

            std::string line;

            ebt::json::json_parser<std::unordered_map<std::string, double>> parser;
            std::unordered_map<std::string, double> mean = parser.parse(ifs);
            std::getline(ifs, line);
            
            std::unordered_map<std::string, double> var = parser.parse(ifs);
            std::getline(ifs, line);

            return std::make_tuple(mean, var);
        }

        void quad_length::operator()(
            feat_t& feat,
            fst::composed_fst<lattice::fst, lm::fst> const& fst,
            std::tuple<int, int> const& e) const
        {
            std::vector<double>& g = lexicalize(order, feat, fst, e);

            double d = fst.fst1->time(std::get<0>(fst.head(e))) - fst.fst1->time(std::get<0>(fst.tail(e)));

            g.push_back(std::pow(d - mean.at(fst.output(e)), 2) / var.at(fst.output(e)));
        }

    }

    namespace first_order {

        composite_feature::composite_feature()
        {}

        void composite_feature::operator()(
            param_t& feat, ilat::fst const& fst, int e) const
        {
            for (auto& f: features) {
                (*f)(feat, fst, e);
            }
        }
        
        la::vector<double>& lexicalize(feat_dim_alloc const& alloc,
            int order, param_t& feat, ilat::fst const& fst, int e)
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

            auto& g = feat.class_vec[label_tuple];
            g.resize(alloc.order_dim[order]);

            return g;
        }

        segment_feature::segment_feature(
            feat_dim_alloc& alloc,
            int order,
            std::shared_ptr<segfeat::la::feature> feat_func,
            std::vector<std::vector<real>> const& frames)
            : alloc(alloc), order(order), feat_func(feat_func), frames(frames)
        {
            dim = alloc.alloc(order, feat_func->dim(frames.front().size()));
        }

        void segment_feature::operator()(
            param_t& feat, ilat::fst const& fst, int e) const
        {
            la::vector<double>& g = lexicalize(alloc, order, feat, fst, e);

            (*feat_func)(dim, g, frames, fst.time(fst.tail(e)), fst.time(fst.head(e)));
        }

        namespace feature {

            lattice_score::lattice_score(feat_dim_alloc& alloc)
                : alloc(alloc)
            {
                dim = alloc.alloc(0, 1);
            }

            void lattice_score::operator()(
                param_t& feat, ilat::fst const& fst, int e) const
            {
                if (feat.class_vec.size() == 0) {
                    feat.class_vec.resize(1);
                    feat.class_vec[0].resize(alloc.order_dim[0]);
                }
                feat.class_vec[0](dim) = fst.weight(e);
            }

            external_feature::external_feature(feat_dim_alloc& alloc,
                    int order, std::vector<int> dims)
                : alloc(alloc), order(order), dims(dims)
            {
                dim = alloc.alloc(order, dims.size());
            }

            void external_feature::operator()(
                param_t& feat, ilat::fst const& fst, int e) const
            {
                auto& g = lexicalize(alloc, order, feat, fst, e);

                std::vector<double> const& f = fst.data->feats.at(e);

                for (int i = 0; i < dims.size(); ++i) {
                    g(dim + i) = f[dims[i]];
                }
            }

            frame_feature::frame_feature(feat_dim_alloc& alloc,
                    std::vector<std::vector<double>> const& frames,
                    std::unordered_map<std::string, std::string> const& args)
                : alloc(alloc), frames(frames)
            {
                dim = alloc.alloc(0, 1);

                if (!ebt::in(std::string("label-dim"), args)) {
                    std::cerr << "--label-dim missing" << std::endl;
                    exit(1);
                }

                if (!ebt::in(std::string("label"), args)) {
                    std::cerr << "--label missing" << std::endl;
                    exit(1);
                }

                label_id = scrf::load_label_id(args.at("label"));
                id_dim = load_label_dim(args.at("label-dim"), label_id);
            }
            
            std::vector<std::vector<int>>
            frame_feature::load_label_dim(std::string filename,
                std::unordered_map<std::string, int> const& label_id)
            {
                std::vector<std::vector<int>> result;
                result.resize(label_id.size());

                std::string line;
                std::ifstream ifs { filename };
            
                while (std::getline(ifs, line)) {
                    auto parts = ebt::split(line);
                    std::vector<int> dims;
                    for (int i = 1; i < parts.size(); ++i) {
                        dims.push_back(std::stoi(parts[i]));
                    }
                    result[label_id.at(parts[0])] = dims;
                }
            
                return result;
            }

            void frame_feature::operator()(
                param_t& feat, ilat::fst const& fst, int e) const
            {
                double result = 0;

                int start = std::min<int>(fst.time(fst.tail(e)), frames.size() - 1);
                int end = std::min<int>(fst.time(fst.head(e)), frames.size());

                for (int i = start; i < end; ++i) {
                    for (int f_dim: id_dim[fst.output(e)]) {
                        result += frames[i][f_dim];
                    }
                }

                if (feat.class_vec.size() == 0) {
                    feat.class_vec.resize(1);
                    feat.class_vec[0].resize(alloc.order_dim[0]);
                }
                feat.class_vec[0](dim) = result;

            }

            quad_length::quad_length(feat_dim_alloc& alloc,
                int order,
                std::unordered_map<std::string, std::string> const& args)
                : alloc(alloc), order(order)
            {
                if (!ebt::in(std::string("length-stat"), args)) {
                    std::cerr << "--length-stat missing" << std::endl;
                    exit(1);
                }

                if (!ebt::in(std::string("label"), args)) {
                    std::cerr << "--label missing" << std::endl;
                    exit(1);
                }

                label_id = scrf::load_label_id(args.at("label"));
                std::tie(mean, var) = load_length_stat(args.at("length-stat"), label_id);

                sils.push_back(label_id.at("<s>"));
                sils.push_back(label_id.at("</s>"));
                sils.push_back(label_id.at("sil"));

                dim = alloc.alloc(order, 1);
            }

            std::tuple<std::vector<double>, std::vector<double>>
            quad_length::load_length_stat(std::string filename,
                std::unordered_map<std::string, int> const& label_id) const
            {
                std::vector<double> mean;
                mean.resize(label_id.size());
                std::vector<double> var;
                var.resize(label_id.size());

                std::ifstream ifs { filename };

                std::string line;

                ebt::json::json_parser<std::unordered_map<std::string, double>> parser;
                for(auto& p: parser.parse(ifs)) {
                    mean[label_id.at(p.first)] = p.second;
                }
                std::getline(ifs, line);
                
                for(auto& p: parser.parse(ifs)) {
                    var[label_id.at(p.first)] = p.second;
                }
                std::getline(ifs, line);

                return std::make_tuple(mean, var);
            }

            void quad_length::operator()(
                param_t& feat, ilat::fst const& fst, int e) const
            {
                la::vector<double>& g = lexicalize(alloc, order, feat, fst, e);

                double d = fst.time(fst.head(e)) - fst.time(fst.tail(e));

                for (int i: sils) {
                    if (fst.output(e) == i) {
                        g(dim) = 0;
                        return;
                    }
                }

                g(dim) = std::pow(d - mean[fst.output(e)], 2) / var[fst.output(e)];
            }

            max_hits::max_hits(feat_dim_alloc& alloc, int order, double percentile,
                int nhits, std::vector<std::vector<double>> const& frames)
                : alloc(alloc), order(order), percentile(percentile), nhits(nhits), frames(frames)
            {
                double inf = std::numeric_limits<double>::infinity();
                max.resize(frames.size());
                min.resize(frames.size());

                for (int i = 0; i < frames.size(); ++i) {
                    max[i] = -inf;
                    min[i] = inf;

                    for (int j = 0; j < frames[i].size(); ++j) {
                        if (frames[i][j] > max[i]) {
                            max[i] = frames[i][j];
                        }

                        if (frames[i][j] < min[i]) {
                            min[i] = frames[i][j];
                        }
                    }
                }

                dim = alloc.alloc(order, nhits + 1);
            }

            void max_hits::operator()(
                param_t& feat, ilat::fst const& fst, int e) const
            {
                la::vector<double>& g = lexicalize(alloc, order, feat, fst, e);

                int ell = fst.output(e);

                int start = std::min<int>(fst.time(fst.tail(e)), frames.size() - 1);
                int end = std::min<int>(fst.time(fst.head(e)), frames.size());

                int hit = 0;
                for (int i = start; i < end; ++i) {
                    if (frames[i][ell] > percentile * max[i] + (1 - percentile) * min[i]) {
                        ++hit;
                    }
                }

                hit = std::min(hit, nhits);

                g(dim + hit) = 1;
            }
        }
    }

    namespace experimental {

        feat_dim_alloc::feat_dim_alloc(std::vector<int> const& labels)
            : labels(labels)
        {}

        int feat_dim_alloc::alloc(int order, int dim)
        {
            if (order >= order_dim.size()) {
                order_dim.resize(order + 1);
            }

            int result = order_dim[order];
            order_dim[order] += dim;

            return result;
        }

    }
}
