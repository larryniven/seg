#include "scrf/weiran.h"
#include <fstream>
#include "ebt/ebt.h"
#include "speech/speech.h"

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
        std::vector<real> v;

        auto h = autodiff::var();
        v.resize(param.weight[0].front().size(), 0);
        h->output = std::make_shared<std::vector<real>>(std::move(v));
        nn.layers.push_back(h);

        auto f = autodiff::var();
        v = std::vector<real>();
        v.resize(param.weight[0].size(), 0);
        f->output = std::make_shared<std::vector<real>>(std::move(v));

        h = autodiff::relu(autodiff::add(
            autodiff::mult(autodiff::var(param.weight[0]), h), f));
        v = std::vector<real>();
        v.resize(param.weight[0].size(), 0);
        v.push_back(1);
        h->output = std::make_shared<std::vector<real>>(std::move(v));
        nn.layers.push_back(h);

        // std::cout << param.weight[0].size() << " " << param.weight[0].front().size() << std::endl;
        // std::cout << param.weight.back().size() << " " << param.weight.back().front().size() << std::endl;

        h = autodiff::logsoftmax(autodiff::mult(autodiff::var(param.weight.back()), h));
        v = std::vector<real>();
        v.resize(param.weight.back().size(), 0);
        h->output = std::make_shared<std::vector<real>>(std::move(v));
        nn.layers.push_back(h);

        /*
        std::cout << "make_nn" << std::endl;
        for (int i = 0; i < nn.layers.size(); ++i) {
            std::cout << i << " " << autodiff::get_output<std::vector<real>>(nn.layers.at(i)).size() << std::endl;
        }
        */

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
        nn_t nn,
        int start_dim,
        int end_dim)
        : frames(frames), cm_mean(cm_mean), cm_stddev(cm_stddev), nn(nn), start_dim(start_dim), end_dim(end_dim)
    {
        if (this->start_dim == -1) {
            this->start_dim = 0;
        }
        if (this->end_dim == -1) {
            this->end_dim = frames.front().size() - 1;
        }
    }

    void weiran_feature::operator()(
        scrf::param_t& feat,
        fst::composed_fst<lattice::fst, lm::fst> const& fst,
        std::tuple<int, int> const& e) const
    {
        if (!ebt::in(std::get<0>(e), feat_cache)) {
            int tail = std::get<0>(fst.tail(e));
            int head = std::get<0>(fst.head(e));

            int tail_time = std::min<int>(frames.size() - 1, std::max<int>(0, fst.fst1->data->vertices.at(tail).time));
            int head_time = std::min<int>(frames.size() - 1, std::max<int>(0, fst.fst1->data->vertices.at(head).time));

            std::vector<real> cm_feat = speech::clarkson_moreno_feature(frames, tail_time,
                head_time, start_dim, start_dim + 38);

            for (int i = 0; i < cm_mean.size(); ++i) {
                cm_feat[i] = (cm_feat[i] - cm_mean[i]) / cm_stddev[i];
            }

            nn.layers.at(0)->output = std::make_shared<std::vector<real>>(std::move(cm_feat));

            auto& f = frames.at((tail_time + head_time) / 2);
            std::vector<real> cnn_mult_cache { f.begin() + 39 + start_dim, f.begin() + end_dim + 1 };
            nn.layers.at(1)->children.at(0)->children.at(1)->output
                = std::make_shared<std::vector<real>>(std::move(cnn_mult_cache));

            autodiff::eval(nn.layers.back(), autodiff::eval_funcs);

            feat_cache[std::get<0>(e)] = autodiff::get_output<std::vector<real>>(nn.layers.back());
        }

        auto& u = feat.class_param["[lattice] " + fst.output(e)];
        auto& v = feat_cache.at(std::get<0>(e));

        u.insert(u.end(), v.begin(), v.end());
    }

}
