#include "scrf/weiran.h"
#include <fstream>
#include "ebt/ebt.h"
#include "speech-util/speech.h"

namespace weiran {

    param_t load_param(std::istream& is)
    {
        param_t param;
        std::string line;

        ebt::json::json_parser<decltype(param.weight)> weight_parser;
        param.weight = weight_parser.parse(is);
        std::getline(is, line);

        return param;
    }

    param_t load_param(std::string filename)
    {
        std::ifstream ifs { filename };

        return load_param(ifs);
    }

    void save_param(param_t const& param, std::ostream& os)
    {
        os << param.weight << std::endl;
    }
    
    void save_param(param_t const& param, std::string filename)
    {
        std::ofstream ofs { filename };
        save_param(param, ofs);
    }

    nn_t make_nn(param_t const& param)
    {
        nn_t nn;

        auto h = autodiff::var();
        std::vector<real> v;
        v.resize(param.weight[0].front().size());
        h->output = std::make_shared<std::vector<real>>(std::move(v));
        nn.layers.push_back(h);

        for (int i = 0; i < param.weight.size() - 1; ++i) {
            h = autodiff::logistic(autodiff::mult(autodiff::var(param.weight[i]), h));
            std::vector<real> v;
            v.resize(param.weight[i].size());
            v.push_back(1);
            h->output = std::make_shared<std::vector<real>>(std::move(v));
            // std::cout << "layer " << i << ": " << param.weight[i].size() << " " << param.weight[i].front().size() << std::endl;
            nn.layers.push_back(h);
        }

        h = autodiff::logsoftmax(autodiff::mult(autodiff::var(param.weight.back()), h));
        // std::cout << "layer " << param.weight.size() - 1 << ": " << param.weight.back().size() << " " << param.weight.back().front().size() << std::endl;
        std::vector<real> w;
        w.resize(param.weight.back().size());
        w.push_back(1);
        h->output = std::make_shared<std::vector<real>>(std::move(w));
        nn.layers.push_back(h);

        return nn;
    }

    void move_in_param(nn_t& nn, param_t& param)
    {
        for (int i = 0; i < param.weight.size(); ++i) {
            auto w = nn.layers[i+1]->children[0]->children[0];
            w->output = std::make_shared<std::vector<std::vector<real>>>(std::move(param.weight[i]));
        }
    }

    void move_out_param(nn_t& nn, param_t& param)
    {
        for (int i = 0; i < param.weight.size(); ++i) {
            auto w = nn.layers[i+1]->children[0]->children[0];
            param.weight[i] = std::move(autodiff::get_output<std::vector<std::vector<real>>>(w));
        }
    }

    weiran_feature::weiran_feature(
        std::vector<std::vector<real>> const& frames,
        std::vector<real> const& cm_mean,
        std::vector<real> const& cm_stddev,
        nn_t nn)
        : frames(frames), cm_mean(cm_mean), cm_stddev(cm_stddev), nn(nn)
    {}

    void weiran_feature::operator()(
        scrf::param_t& feat,
        fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        if (!ebt::in(std::get<0>(e), feat_cache)) {
            int tail = std::get<0>(fst.tail(e));
            int head = std::get<1>(fst.head(e));

            int tail_time = std::min<int>(frames.size() - 1, std::max<int>(0, fst.fst1->data->vertices.at(tail).time));
            int head_time = std::min<int>(frames.size() - 1, std::max<int>(0, fst.fst1->data->vertices.at(head).time));

            auto& f = frames.at((tail_time + head_time) / 2);
            std::vector<real> v { f.begin() + 39, f.end() };
            std::vector<real> cm_feat = speech::clarkson_moreno_feature(frames, tail_time, head_time, 39);

            for (int i = 0; i < cm_mean.size(); ++i) {
                cm_feat[i] = (cm_feat[i] - cm_mean[i]) / cm_stddev[i];
            }

            v.insert(v.end(), cm_feat.begin(), cm_feat.end());

            nn.layers.at(0)->output = std::make_shared<std::vector<real>>(std::move(v));

            autodiff::eval(nn.layers.back(), autodiff::eval_funcs);

            /*
            for (int i = 0; i < nn.layers.size(); ++i) {
                std::cout << "hidden " << i << ": " << autodiff::get_output<std::vector<real>>(nn.layers[i]).size() << std::endl;
            }
            */

            feat_cache[std::get<0>(e)] = std::move(autodiff::get_output<std::vector<real>>(nn.layers.back()));
        }

        auto& u = feat.class_param["[lattice] " + fst.output(e)];
        auto& v = feat_cache.at(std::get<0>(e));

        u.insert(u.end(), v.begin(), v.end());
    }

}
