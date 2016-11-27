#include "seg/fscrf.h"
#include "seg/scrf_weight.h"
#include "seg/util.h"
#include "seg/scrf.h"
#include "nn/lstm-tensor-tree.h"
#include <fstream>

namespace fscrf {

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

    std::shared_ptr<ilat::fst> make_random_graph(int frames,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label,
        int min_seg_len, int max_seg_len, int stride,
        double prob,
        std::default_random_engine& gen)
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

        std::bernoulli_distribution dist { prob };

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

                    if (!dist(gen)) {
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

    std::shared_ptr<tensor_tree::vertex> make_tensor_tree(
        std::vector<std::string> const& features)
    {
        tensor_tree::vertex root { tensor_tree::tensor_t::nil };

        for (auto& k: features) {
            if (ebt::startswith(k, "ext0")) {
                root.children.push_back(tensor_tree::make_vector("ext0"));
            } else if (ebt::startswith(k, "ext1")) {
                root.children.push_back(tensor_tree::make_matrix("ext1"));
            } else if (ebt::startswith(k, "frame-avg")) {
                root.children.push_back(tensor_tree::make_matrix("frame avg"));
            } else if (ebt::startswith(k, "frame-att")) {
                root.children.push_back(tensor_tree::make_matrix("frame att"));
                root.children.push_back(tensor_tree::make_matrix("frame att"));
            } else if (ebt::startswith(k, "frame-samples")) {
                root.children.push_back(tensor_tree::make_matrix("frame samples"));
                root.children.push_back(tensor_tree::make_matrix("frame samples"));
                root.children.push_back(tensor_tree::make_matrix("frame samples"));
            } else if (ebt::startswith(k, "left-boundary")) {
                root.children.push_back(tensor_tree::make_matrix("left boundary"));
                root.children.push_back(tensor_tree::make_matrix("left boundary"));
                root.children.push_back(tensor_tree::make_matrix("left boundary"));
            } else if (ebt::startswith(k, "right-boundary")) {
                root.children.push_back(tensor_tree::make_matrix("right boundary"));
                root.children.push_back(tensor_tree::make_matrix("right boundary"));
                root.children.push_back(tensor_tree::make_matrix("right boundary"));
            } else if (ebt::startswith(k, "length-indicator")) {
                root.children.push_back(tensor_tree::make_matrix("length"));
            } else if (k == "lm") {
                root.children.push_back(tensor_tree::make_vector("lm"));
            } else if (k == "lat-weight") {
                root.children.push_back(tensor_tree::make_vector("lat-weight"));
            } else if (k == "bias0") {
                root.children.push_back(tensor_tree::make_vector("bias0"));
            } else if (k == "bias1") {
                root.children.push_back(tensor_tree::make_vector("bias1"));
            } else if (k == "boundary2") {
                tensor_tree::vertex v { tensor_tree::tensor_t::nil };
                v.children.push_back(tensor_tree::make_matrix("left boundary order2 acoustic embedding"));
                v.children.push_back(tensor_tree::make_matrix("left boundary order2 label1 embedding"));
                v.children.push_back(tensor_tree::make_matrix("left boundary order2 label2 embedding"));
                v.children.push_back(tensor_tree::make_matrix("left boundary order2 weight"));
                v.children.push_back(tensor_tree::make_vector("left boundary order2 weight"));
                root.children.push_back(std::make_shared<tensor_tree::vertex>(v));
            } else if (k == "segrnn") {
                tensor_tree::vertex v { tensor_tree::tensor_t::nil };
                v.children.push_back(tensor_tree::make_matrix("segrnn left embedding"));
                v.children.push_back(tensor_tree::make_vector("segrnn left end"));
                v.children.push_back(tensor_tree::make_matrix("segrnn right embedding"));
                v.children.push_back(tensor_tree::make_vector("segrnn right end"));
                v.children.push_back(tensor_tree::make_matrix("segrnn label embedding"));
                v.children.push_back(tensor_tree::make_matrix("segrnn label embedding"));
                v.children.push_back(tensor_tree::make_matrix("segrnn length embedding"));
                v.children.push_back(tensor_tree::make_matrix("segrnn length embedding"));
                v.children.push_back(tensor_tree::make_vector("segrnn bias1"));
                v.children.push_back(tensor_tree::make_matrix("segrnn weight1"));
                v.children.push_back(tensor_tree::make_vector("segrnn bias2"));
                v.children.push_back(tensor_tree::make_vector("segrnn weight2"));
                root.children.push_back(std::make_shared<tensor_tree::vertex>(v));
            } else if (k == "segrnn-mod") {
                tensor_tree::vertex v { tensor_tree::tensor_t::nil };
                v.children.push_back(tensor_tree::make_matrix("segrnn left embedding"));
                v.children.push_back(tensor_tree::make_matrix("segrnn right embedding"));
                v.children.push_back(tensor_tree::make_matrix("segrnn label embedding"));
                v.children.push_back(tensor_tree::make_matrix("segrnn label embedding"));
                v.children.push_back(tensor_tree::make_matrix("segrnn length embedding"));
                v.children.push_back(tensor_tree::make_matrix("segrnn length embedding"));
                v.children.push_back(tensor_tree::make_vector("segrnn bias1"));
                v.children.push_back(tensor_tree::make_matrix("segrnn weight1"));
                v.children.push_back(tensor_tree::make_vector("segrnn bias2"));
                v.children.push_back(tensor_tree::make_vector("segrnn weight2"));
                v.children.push_back(tensor_tree::make_vector("segrnn bias3"));
                root.children.push_back(std::make_shared<tensor_tree::vertex>(v));
            } else {
                std::cout << "unknown feature " << k << std::endl;
                exit(1);
            }
        }

        return std::make_shared<tensor_tree::vertex>(root);
    }

    std::shared_ptr<scrf::composite_weight<ilat::fst>> make_weights(
        std::vector<std::string> const& features,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::shared_ptr<autodiff::op_t> frame_mat,
        double dropout,
        std::default_random_engine *gen)
    {
        scrf::composite_weight<ilat::fst> weight_func;

        int feat_idx = 0;

        for (auto& k: features) {
            if (ebt::startswith(k, "frame-avg")) {
                weight_func.weights.push_back(std::make_shared<fscrf::frame_avg_score>(
                    fscrf::frame_avg_score(tensor_tree::get_var(var_tree->children[feat_idx]), frame_mat)));

                ++feat_idx;
            } else if (ebt::startswith(k, "frame-att")) {
                weight_func.weights.push_back(std::make_shared<fscrf::frame_weighted_avg_score>(
                    fscrf::frame_weighted_avg_score(tensor_tree::get_var(var_tree->children[feat_idx]),
                        tensor_tree::get_var(var_tree->children[feat_idx + 1]),
                        frame_mat)));

                feat_idx += 2;
            } else if (ebt::startswith(k, "frame-samples")) {
                weight_func.weights.push_back(std::make_shared<fscrf::frame_samples_score>(
                    fscrf::frame_samples_score(tensor_tree::get_var(var_tree->children[feat_idx]), frame_mat, 1.0 / 6)));
                weight_func.weights.push_back(std::make_shared<fscrf::frame_samples_score>(
                    fscrf::frame_samples_score(tensor_tree::get_var(var_tree->children[feat_idx + 1]), frame_mat, 1.0 / 2)));
                weight_func.weights.push_back(std::make_shared<fscrf::frame_samples_score>(
                    fscrf::frame_samples_score(tensor_tree::get_var(var_tree->children[feat_idx + 2]), frame_mat, 5.0 / 6)));

                feat_idx += 3;
            } else if (ebt::startswith(k, "left-boundary")) {
                weight_func.weights.push_back(std::make_shared<fscrf::left_boundary_score>(
                    fscrf::left_boundary_score(tensor_tree::get_var(var_tree->children[feat_idx]), frame_mat, -1)));
                weight_func.weights.push_back(std::make_shared<fscrf::left_boundary_score>(
                    fscrf::left_boundary_score(tensor_tree::get_var(var_tree->children[feat_idx + 1]), frame_mat, -2)));
                weight_func.weights.push_back(std::make_shared<fscrf::left_boundary_score>(
                    fscrf::left_boundary_score(tensor_tree::get_var(var_tree->children[feat_idx + 2]), frame_mat, -3)));

                feat_idx += 3;
            } else if (ebt::startswith(k, "right-boundary")) {
                weight_func.weights.push_back(std::make_shared<fscrf::right_boundary_score>(
                    fscrf::right_boundary_score(tensor_tree::get_var(var_tree->children[feat_idx]), frame_mat, 1)));
                weight_func.weights.push_back(std::make_shared<fscrf::right_boundary_score>(
                    fscrf::right_boundary_score(tensor_tree::get_var(var_tree->children[feat_idx + 1]), frame_mat, 2)));
                weight_func.weights.push_back(std::make_shared<fscrf::right_boundary_score>(
                    fscrf::right_boundary_score(tensor_tree::get_var(var_tree->children[feat_idx + 2]), frame_mat, 3)));

                feat_idx += 3;
            } else if (ebt::startswith(k, "length-indicator")) {
                weight_func.weights.push_back(std::make_shared<fscrf::length_score>(
                    fscrf::length_score { tensor_tree::get_var(var_tree->children[feat_idx]) }));

                ++feat_idx;
            } else if (ebt::startswith(k, "ext0")) {
                auto parts = ebt::split(k, ":");
                parts = ebt::split(parts[1], "+");
                std::vector<int> dims;

                for (auto& p: parts) {
                    std::vector<std::string> range = ebt::split(p, "-");
                    if (range.size() == 2) {
                        for (int i = std::stoi(range[0]); i <= std::stoi(range[1]); ++i) {
                            dims.push_back(i);
                        }
                    } else if (range.size() == 1) {
                        dims.push_back(std::stoi(p));
                    } else {
                        std::cerr << "unknown external feature format: " << k << std::endl;
                    }
                }

                weight_func.weights.push_back(std::make_shared<fscrf::external_score_order0>(
                    fscrf::external_score_order0 { tensor_tree::get_var(var_tree->children[feat_idx]), dims }));

                ++feat_idx;
            } else if (ebt::startswith(k, "ext1")) {
                auto parts = ebt::split(k, ":");
                parts = ebt::split(parts[1], "+");
                std::vector<int> dims;

                for (auto& p: parts) {
                    std::vector<std::string> range = ebt::split(p, "-");
                    if (range.size() == 2) {
                        for (int i = std::stoi(range[0]); i <= std::stoi(range[1]); ++i) {
                            dims.push_back(i);
                        }
                    } else if (range.size() == 1) {
                        dims.push_back(std::stoi(p));
                    } else {
                        std::cerr << "unknown external feature format: " << k << std::endl;
                    }
                }

                weight_func.weights.push_back(std::make_shared<fscrf::external_score_order1>(
                    fscrf::external_score_order1 { tensor_tree::get_var(var_tree->children[feat_idx]), dims }));

                ++feat_idx;
            } else if (k == "bias0") {
                weight_func.weights.push_back(std::make_shared<fscrf::bias0_score>(
                    fscrf::bias0_score { tensor_tree::get_var(var_tree->children[feat_idx]) }));

                ++feat_idx;
            } else if (k == "bias1") {
                weight_func.weights.push_back(std::make_shared<fscrf::bias1_score>(
                    fscrf::bias1_score { tensor_tree::get_var(var_tree->children[feat_idx]) }));

                ++feat_idx;
            } else if (k == "lat-weight") {
                weight_func.weights.push_back(std::make_shared<edge_weight>(edge_weight {
                    tensor_tree::get_var(var_tree->children[feat_idx]) }));

                ++feat_idx;
            } else if (ebt::startswith(k, "segrnn-mod")) {
                weight_func.weights.push_back(std::make_shared<fscrf::segrnn_mod_score>(
                    fscrf::segrnn_mod_score(var_tree->children[feat_idx], frame_mat, dropout, gen)));

                ++feat_idx;
            } else if (ebt::startswith(k, "segrnn")) {
                weight_func.weights.push_back(std::make_shared<fscrf::segrnn_score>(
                    fscrf::segrnn_score(var_tree->children[feat_idx], frame_mat, dropout, gen)));

                ++feat_idx;
            } else {
                std::cout << "unknown feature: " << k << std::endl;
                exit(1);
            }
        }

        return std::make_shared<scrf::composite_weight<ilat::fst>>(weight_func);
    }

    frame_avg_score::frame_avg_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames)
        : param(param), frames(frames)
    {
        score = autodiff::rtmul(param, frames);
        autodiff::eval_vertex(score, autodiff::eval_funcs);
    }

    double frame_avg_score::operator()(ilat::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::matrix<double>>(score);

        double sum = 0;

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        for (int t = tail_time; t < head_time; ++t) {
            sum += m(ell, t);
        }

        return (head_time <= tail_time ? 0 : sum / (head_time - tail_time));
    }

    void frame_avg_score::accumulate_grad(double g, ilat::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::matrix<double>>(score);

        if (score->grad == nullptr) {
            la::matrix<double> m_grad;
            m_grad.resize(m.rows(), m.cols());
            score->grad = std::make_shared<la::matrix<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::matrix<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        for (int t = tail_time; t < head_time; ++t) {
            m_grad(ell, t) += g / (head_time - tail_time);
        }
    }

    void frame_avg_score::grad() const
    {
        autodiff::eval_vertex(score, autodiff::grad_funcs);
    }

    frame_weighted_avg_score::frame_weighted_avg_score(
            std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> att_param,
            std::shared_ptr<autodiff::op_t> frames)
        : param(param), att_param(att_param), frames(frames)
    {
        score = autodiff::rtmul(param, frames);
        autodiff::eval_vertex(score, autodiff::eval_funcs);

        att = autodiff::rtmul(att_param, frames);
        att_exp = autodiff::mexp(att);
        autodiff::eval_vertex(att, autodiff::eval_funcs);
        autodiff::eval_vertex(att_exp, autodiff::eval_funcs);
    }

    double frame_weighted_avg_score::operator()(ilat::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::matrix<double>>(score);
        auto& n = autodiff::get_output<la::matrix<double>>(att_exp);

        double sum = 0;

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        int start = std::max<int>(0, tail_time);
        int end = std::min<int>(head_time, n.cols());

        double Z = 0;
        for (int t = start; t < end; ++t) {
            Z += n(ell, t) + 0.01;
        }

        for (int t = start; t < end; ++t) {
            sum += (n(ell, t) + 0.01) * m(ell, t) / Z;
        }

        return sum;
    }

    void frame_weighted_avg_score::accumulate_grad(double g, ilat::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::matrix<double>>(score);
        auto& n = autodiff::get_output<la::matrix<double>>(att_exp);

        if (score->grad == nullptr) {
            la::matrix<double> m_grad;
            m_grad.resize(m.rows(), m.cols());
            score->grad = std::make_shared<la::matrix<double>>(std::move(m_grad));
        }

        if (att_exp->grad == nullptr) {
            la::matrix<double> n_grad;
            n_grad.resize(n.rows(), n.cols());
            att_exp->grad = std::make_shared<la::matrix<double>>(std::move(n_grad));
        }

        auto& m_grad = autodiff::get_grad<la::matrix<double>>(score);
        auto& n_grad = autodiff::get_grad<la::matrix<double>>(att_exp);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        int start = std::max<int>(0, tail_time);
        int end = std::min<int>(head_time, n.cols());

        double Z = 0;
        for (int t = start; t < end; ++t) {
            Z += n(ell, t) + 0.01;
        }

        double sum = 0;
        for (int t = start; t < end; ++t) {
            sum += (n(ell, t) + 0.01) * m(ell, t) / Z;
        }

        for (int t = start; t < end; ++t) {
            m_grad(ell, t) += g * (n(ell, t) + 0.01) / Z;
            n_grad(ell, t) += g * (m(ell, t) - sum) / Z;

            if (std::isnan(m_grad(ell, t))) {
                std::cout << "m_grad has nan" << std::endl;
            }

            if (std::isnan(n_grad(ell, t))) {
                std::cout << "n_grad has nan" << std::endl;
            }
        }
    }

    void frame_weighted_avg_score::grad() const
    {
        autodiff::eval_vertex(score, autodiff::grad_funcs);
        autodiff::eval_vertex(att_exp, autodiff::grad_funcs);
        autodiff::eval_vertex(att, autodiff::grad_funcs);
    }

    frame_samples_score::frame_samples_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames, double scale)
        : param(param), frames(frames), scale(scale)
    {
        score = autodiff::rtmul(param, frames);
        autodiff::eval_vertex(score, autodiff::eval_funcs);
    }

    double frame_samples_score::operator()(ilat::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::matrix<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        int dur = head_time - tail_time;

        return m(ell, tail_time + dur * scale);
    }

    void frame_samples_score::accumulate_grad(double g, ilat::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::matrix<double>>(score);

        if (score->grad == nullptr) {
            la::matrix<double> m_grad;
            m_grad.resize(m.rows(), m.cols());
            score->grad = std::make_shared<la::matrix<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::matrix<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        int dur = head_time - tail_time;

        m_grad(ell, tail_time + dur * scale) += g;
    }

    void frame_samples_score::grad() const
    {
        autodiff::eval_vertex(score, autodiff::grad_funcs);
    }

    left_boundary_score::left_boundary_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames, int shift)
        : param(param), frames(frames), shift(shift)
    {
        score = autodiff::rtmul(param, frames);
        autodiff::eval_vertex(score, autodiff::eval_funcs);
    }

    double left_boundary_score::operator()(ilat::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::matrix<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        return m(ell, std::max<int>(tail_time + shift, 0));
    }

    void left_boundary_score::accumulate_grad(double g, ilat::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::matrix<double>>(score);

        if (score->grad == nullptr) {
            la::matrix<double> m_grad;
            m_grad.resize(m.rows(), m.cols());
            score->grad = std::make_shared<la::matrix<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::matrix<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        m_grad(ell, std::max<int>(tail_time + shift, 0)) += g;
    }

    void left_boundary_score::grad() const
    {
        autodiff::eval_vertex(score, autodiff::grad_funcs);
    }

    right_boundary_score::right_boundary_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames, int shift)
        : param(param), frames(frames), shift(shift)
    {
        score = autodiff::rtmul(param, frames);
        autodiff::eval_vertex(score, autodiff::eval_funcs);
    }

    double right_boundary_score::operator()(ilat::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::matrix<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        return m(ell, std::min<int>(head_time + shift, m.cols() - 1));
    }

    void right_boundary_score::accumulate_grad(double g, ilat::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::matrix<double>>(score);

        if (score->grad == nullptr) {
            la::matrix<double> m_grad;
            m_grad.resize(m.rows(), m.cols());
            score->grad = std::make_shared<la::matrix<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::matrix<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        m_grad(ell, std::min<int>(head_time + shift, m.cols() - 1)) += g;
    }

    void right_boundary_score::grad() const
    {
        autodiff::eval_vertex(score, autodiff::grad_funcs);
    }

    segrnn_mod_score::segrnn_mod_score(std::shared_ptr<tensor_tree::vertex> param,
        std::shared_ptr<autodiff::op_t> frames)
        : segrnn_mod_score(param, frames, 0.0, nullptr)
    {
    }

    segrnn_mod_score::segrnn_mod_score(std::shared_ptr<tensor_tree::vertex> param,
        std::shared_ptr<autodiff::op_t> frames,
        double dropout,
        std::default_random_engine *gen)
        : param(param), frames(frames), dropout(dropout), gen(gen)
    {
        assert(0.0 <= dropout && dropout <= 1.0);

        autodiff::computation_graph& graph = *frames->graph;

        std::shared_ptr<autodiff::op_t> frames_tmp = graph.var();

        pre_left = autodiff::mmul(frames_tmp, tensor_tree::get_var(param->children[0]));
        pre_right = autodiff::mmul(frames_tmp, tensor_tree::get_var(param->children[1]));
        pre_label = autodiff::mmul(tensor_tree::get_var(param->children[2]),
            tensor_tree::get_var(param->children[3]));
        pre_length = autodiff::mmul(tensor_tree::get_var(param->children[4]),
            tensor_tree::get_var(param->children[5]));

        std::unordered_set<std::shared_ptr<autodiff::op_t>> exclude {
            pre_left, pre_right, pre_label, pre_length, frames_tmp };

        for (int i = 0; i <= 9; ++i) {
            exclude.insert(tensor_tree::get_var(param->children[i]));
        }

        auto left_embedding = autodiff::row_at(pre_left, 0);
        auto right_embedding = autodiff::row_at(pre_right, 0);
        auto label_embedding = autodiff::row_at(pre_label, 0);
        auto length_embedding = autodiff::row_at(pre_length, 0);
        auto mask = graph.var();

        score = autodiff::dot(tensor_tree::get_var(param->children[9]),
            autodiff::emul(mask,
                autodiff::tanh(
                    autodiff::add(
                        tensor_tree::get_var(param->children[8]),
                        autodiff::mul(
                            tensor_tree::get_var(param->children[7]),
                            autodiff::relu(
                                autodiff::add(std::vector<std::shared_ptr<autodiff::op_t>> {
                                    left_embedding,
                                    right_embedding,
                                    label_embedding,
                                    length_embedding,
                                    tensor_tree::get_var(param->children[6])
                                })
                            )
                        )
                    )
                )
            ));

        std::vector<std::shared_ptr<autodiff::op_t>> topo_order_tmp = autodiff::topo_order(score);

        for (auto& i: topo_order_tmp) {
            if (ebt::in(i, exclude)) {
                continue;
            }

            topo_order_shift.push_back(i->id);
        }

        graph.adj[pre_left->id][0] = frames->id;
        graph.adj[pre_right->id][0] = frames->id;

        autodiff::eval_vertex(pre_left, autodiff::eval_funcs);
        autodiff::eval_vertex(pre_right, autodiff::eval_funcs);
        autodiff::eval_vertex(pre_label, autodiff::eval_funcs);
        autodiff::eval_vertex(pre_length, autodiff::eval_funcs);
    }

    double segrnn_mod_score::operator()(ilat::fst const& f,
        int e) const
    {
        if (edge_scores.size() > e && edge_scores[e] != nullptr && edge_scores[e]->output != nullptr) {
            return autodiff::get_output<double>(edge_scores[e]);
        }

        autodiff::computation_graph& comp_graph = *frames->graph;

        auto& m = autodiff::get_output<la::matrix<double>>(frames);
        auto& length_param = autodiff::get_output<la::matrix<double>>(
            tensor_tree::get_var(param->children[4]));

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        auto left_embedding = autodiff::row_at(pre_left, std::max<int>(0, tail_time));
        auto right_embedding = autodiff::row_at(pre_right, std::min<int>(m.rows() - 1, head_time));
        auto label_embedding = autodiff::row_at(pre_label, ell);
        auto length_embedding = autodiff::row_at(pre_length,
            std::min<int>(int(std::log(head_time - tail_time) / std::log(1.6)) + 1,
                 length_param.rows() - 1));

        auto& theta = autodiff::get_output<la::vector<double>>(tensor_tree::get_var(param->children[10]));
        la::vector<double> mask_vec;

        if (dropout == 0.0) {
            mask_vec.resize(theta.size(), 1);
        } else {
            mask_vec.resize(theta.size());
            std::bernoulli_distribution dist {1 - dropout};

            for (int i = 0; i < mask_vec.size(); ++i) {
                mask_vec(i) = dist(*gen) / (1.0 - dropout);
            }
        }

        auto mask = comp_graph.var(mask_vec);

        std::shared_ptr<autodiff::op_t> s_e = autodiff::dot(tensor_tree::get_var(param->children[9]),
            autodiff::emul(mask,
                autodiff::tanh(
                    autodiff::add(
                        tensor_tree::get_var(param->children[8]),
                        autodiff::mul(
                            tensor_tree::get_var(param->children[7]),
                            autodiff::relu(
                                autodiff::add(std::vector<std::shared_ptr<autodiff::op_t>> {
                                    left_embedding,
                                    right_embedding,
                                    label_embedding,
                                    length_embedding,
                                    tensor_tree::get_var(param->children[6])
                                })
                            )
                        )
                    )
                )
            ));

        if (e >= edge_scores.size()) {
            edge_scores.resize(e + 1, nullptr);
        }

        edge_scores[e] = s_e;

        std::vector<std::shared_ptr<autodiff::op_t>> topo_order;
        int d = s_e->id - score->id;
        for (auto& i: topo_order_shift) {
            topo_order.push_back(comp_graph.vertices[i + d]);
        }

        autodiff::eval(topo_order, autodiff::eval_funcs);

        return autodiff::get_output<double>(s_e);
    }

    void segrnn_mod_score::accumulate_grad(double g, ilat::fst const& f,
        int e) const
    {
        std::shared_ptr<autodiff::op_t> s_e = edge_scores[e];
        if (s_e->grad == nullptr) {
            s_e->grad = std::make_shared<double>(0);
        }
        autodiff::get_grad<double>(s_e) += g;
    }

    void segrnn_mod_score::grad() const
    {
        autodiff::computation_graph& comp_graph = *frames->graph;

        for (auto& t: edge_scores) {
            if (t != nullptr) {
                std::vector<std::shared_ptr<autodiff::op_t>> topo_order;
                int d = t->id - score->id;
                for (auto& i: topo_order_shift) {
                    topo_order.push_back(comp_graph.vertices[i + d]);
                }
                autodiff::grad(topo_order, autodiff::grad_funcs);
            }
        }

        autodiff::eval_vertex(pre_left, autodiff::grad_funcs);
        autodiff::eval_vertex(pre_right, autodiff::grad_funcs);
        autodiff::eval_vertex(pre_label, autodiff::grad_funcs);
        autodiff::eval_vertex(pre_length, autodiff::grad_funcs);
    }

    segrnn_score::segrnn_score(std::shared_ptr<tensor_tree::vertex> param,
        std::shared_ptr<autodiff::op_t> frames)
        : segrnn_score(param, frames, 0.0, nullptr)
    {
    }

    segrnn_score::segrnn_score(std::shared_ptr<tensor_tree::vertex> param,
        std::shared_ptr<autodiff::op_t> frames,
        double dropout,
        std::default_random_engine *gen)
        : param(param), frames(frames), dropout(dropout), gen(gen)
    {
        assert(0.0 <= dropout && dropout <= 1.0);

        autodiff::computation_graph& graph = *frames->graph;

        std::shared_ptr<autodiff::op_t> frames_tmp = graph.var();

        left_end = autodiff::lmul(tensor_tree::get_var(param->children[1]), tensor_tree::get_var(param->children[0]));
        right_end = autodiff::lmul(tensor_tree::get_var(param->children[3]), tensor_tree::get_var(param->children[2]));

        pre_left = autodiff::mmul(frames_tmp, tensor_tree::get_var(param->children[0]));
        pre_right = autodiff::mmul(frames_tmp, tensor_tree::get_var(param->children[2]));
        pre_label = autodiff::mmul(tensor_tree::get_var(param->children[4]),
            tensor_tree::get_var(param->children[5]));
        pre_length = autodiff::mmul(tensor_tree::get_var(param->children[6]),
            tensor_tree::get_var(param->children[7]));

        std::unordered_set<std::shared_ptr<autodiff::op_t>> exclude {
            pre_left, pre_right, pre_label, pre_length, left_end, right_end, frames_tmp };

        for (int i = 0; i <= 11; ++i) {
            exclude.insert(tensor_tree::get_var(param->children[i]));
        }

        auto left_embedding = autodiff::row_at(pre_left, 0);
        auto right_embedding = autodiff::row_at(pre_right, 0);
        auto label_embedding = autodiff::row_at(pre_label, 0);
        auto length_embedding = autodiff::row_at(pre_length, 0);
        auto mask = graph.var();

        score = autodiff::dot(tensor_tree::get_var(param->children[11]),
            autodiff::emul(mask,
                autodiff::tanh(
                    autodiff::add(
                        tensor_tree::get_var(param->children[10]),
                        autodiff::mul(
                            tensor_tree::get_var(param->children[9]),
                            autodiff::relu(
                                autodiff::add(std::vector<std::shared_ptr<autodiff::op_t>> {
                                    left_embedding,
                                    right_embedding,
                                    label_embedding,
                                    length_embedding,
                                    tensor_tree::get_var(param->children[8])
                                })
                            )
                        )
                    )
                )
            ));

        std::vector<std::shared_ptr<autodiff::op_t>> topo_order_tmp = autodiff::topo_order(score);

        for (auto& i: topo_order_tmp) {
            if (ebt::in(i, exclude)) {
                continue;
            }

            topo_order_shift.push_back(i->id);
        }

        graph.adj[pre_left->id][0] = frames->id;
        graph.adj[pre_right->id][0] = frames->id;

        autodiff::eval_vertex(pre_left, autodiff::eval_funcs);
        autodiff::eval_vertex(pre_right, autodiff::eval_funcs);
        autodiff::eval_vertex(left_end, autodiff::eval_funcs);
        autodiff::eval_vertex(right_end, autodiff::eval_funcs);
        autodiff::eval_vertex(pre_label, autodiff::eval_funcs);
        autodiff::eval_vertex(pre_length, autodiff::eval_funcs);
    }

    double segrnn_score::operator()(ilat::fst const& f,
        int e) const
    {
        if (edge_scores.size() > e && edge_scores[e] != nullptr && edge_scores[e]->output != nullptr) {
            return autodiff::get_output<double>(edge_scores[e]);
        }

        autodiff::computation_graph& comp_graph = *frames->graph;

        auto& m = autodiff::get_output<la::matrix<double>>(frames);
        auto& length_param = autodiff::get_output<la::matrix<double>>(
            tensor_tree::get_var(param->children[6]));

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        std::shared_ptr<autodiff::op_t> left_embedding;

        if (std::max<int>(0, tail_time) == 0) {
            left_embedding = autodiff::add(std::vector<std::shared_ptr<autodiff::op_t>>{ left_end });
        } else {
            left_embedding = autodiff::row_at(pre_left, std::max<int>(0, tail_time));
        }

        std::shared_ptr<autodiff::op_t> right_embedding;

        if (std::min<int>(m.rows() - 1, head_time) == m.rows() - 1) {
            right_embedding = autodiff::add(std::vector<std::shared_ptr<autodiff::op_t>>{ right_end });
        } else {
            right_embedding = autodiff::row_at(pre_right, std::min<int>(m.rows() - 1, head_time));
        }

        auto label_embedding = autodiff::row_at(pre_label, ell);
        auto length_embedding = autodiff::row_at(pre_length,
            std::min<int>(int(std::log(head_time - tail_time) / std::log(1.6)) + 1,
                 length_param.rows() - 1));

        auto& theta = autodiff::get_output<la::vector<double>>(tensor_tree::get_var(param->children[11]));
        la::vector<double> mask_vec;

        if (dropout == 0.0) {
            mask_vec.resize(theta.size(), 1);
        } else {
            mask_vec.resize(theta.size());
            std::bernoulli_distribution dist {1 - dropout};

            for (int i = 0; i < mask_vec.size(); ++i) {
                mask_vec(i) = dist(*gen) / (1.0 - dropout);
            }
        }

        auto mask = comp_graph.var(mask_vec);

        std::shared_ptr<autodiff::op_t> s_e = autodiff::dot(tensor_tree::get_var(param->children[11]),
            autodiff::emul(mask,
                autodiff::tanh(
                    autodiff::add(
                        tensor_tree::get_var(param->children[10]),
                        autodiff::mul(
                            tensor_tree::get_var(param->children[9]),
                            autodiff::relu(
                                autodiff::add(std::vector<std::shared_ptr<autodiff::op_t>> {
                                    left_embedding,
                                    right_embedding,
                                    label_embedding,
                                    length_embedding,
                                    tensor_tree::get_var(param->children[8])
                                })
                            )
                        )
                    )
                )
            ));

        if (e >= edge_scores.size()) {
            edge_scores.resize(e + 1, nullptr);
        }

        edge_scores[e] = s_e;

        std::vector<std::shared_ptr<autodiff::op_t>> topo_order;
        int d = s_e->id - score->id;
        for (auto& i: topo_order_shift) {
            topo_order.push_back(comp_graph.vertices[i + d]);
        }

        autodiff::eval(topo_order, autodiff::eval_funcs);

        return autodiff::get_output<double>(s_e);
    }

    void segrnn_score::accumulate_grad(double g, ilat::fst const& f,
        int e) const
    {
        std::shared_ptr<autodiff::op_t> s_e = edge_scores[e];
        if (s_e->grad == nullptr) {
            s_e->grad = std::make_shared<double>(0);
        }
        autodiff::get_grad<double>(s_e) += g;
    }

    void segrnn_score::grad() const
    {
        autodiff::computation_graph& comp_graph = *frames->graph;

        for (auto& t: edge_scores) {
            if (t != nullptr) {
                std::vector<std::shared_ptr<autodiff::op_t>> topo_order;
                int d = t->id - score->id;
                for (auto& i: topo_order_shift) {
                    topo_order.push_back(comp_graph.vertices[i + d]);
                }
                autodiff::grad(topo_order, autodiff::grad_funcs);
            }
        }

        autodiff::eval_vertex(pre_left, autodiff::grad_funcs);
        autodiff::eval_vertex(pre_right, autodiff::grad_funcs);
        autodiff::eval_vertex(left_end, autodiff::grad_funcs);
        autodiff::eval_vertex(right_end, autodiff::grad_funcs);
        autodiff::eval_vertex(pre_label, autodiff::grad_funcs);
        autodiff::eval_vertex(pre_length, autodiff::grad_funcs);
    }

    left_boundary_order2_score::left_boundary_order2_score(
            std::shared_ptr<tensor_tree::vertex> param,
            std::vector<std::vector<double>> const& frames,
            int context)
        : param(param), frames(frames), context(context)
    {
        std::vector<double> zero;
        zero.resize(frames[0].size());

        autodiff::computation_graph& graph = *tensor_tree::get_var(param->children[0])->graph;

        for (int i = 0; i < frames.size(); ++i) {
            std::vector<double> v;
            for (int j = i - context; j <= i + context; ++j) {
                if (j < 0 || j >= frames.size()) {
                    v.insert(v.end(), zero.begin(), zero.end());
                } else {
                    v.insert(v.end(), frames[j].begin(), frames[j].end());
                }
            }
            frames_cat.push_back(graph.var(la::vector<double>{v}));
        }

        label_embedding1 = autodiff::row_at(tensor_tree::get_var(param->children[1]), 0);
        label_embedding2 = autodiff::row_at(tensor_tree::get_var(param->children[2]), 0);
        input = graph.var();

        score = autodiff::dot(tensor_tree::get_var(param->children[4]),
            autodiff::relu(autodiff::mul(tensor_tree::get_var(param->children[3]),
            autodiff::relu(autodiff::add(std::vector<std::shared_ptr<autodiff::op_t>> {
                autodiff::mul(tensor_tree::get_var(param->children[0]), input),
                label_embedding1,
                label_embedding2,
            })))));

        std::vector<std::shared_ptr<autodiff::op_t>> tmp_topo_order = autodiff::topo_order(score);

        std::unordered_set<std::shared_ptr<autodiff::op_t>> exclude;
        for (int i = 0; i <= 4; ++i) {
            exclude.insert(tensor_tree::get_var(param->children[i]));
        }

        for (auto& t: tmp_topo_order) {
            if (ebt::in(t, exclude)) {
                continue;
            }

            topo_order.push_back(t);
        }

    }

    double left_boundary_order2_score::operator()(ilat::pair_fst const& f,
        std::tuple<int, int> e) const
    {
        int time = std::min<int>(std::max<int>(0, f.time(f.tail(e))), frames_cat.size() - 1);

        int label1 = f.output(e);
        int label2 = std::get<1>(f.tail(e));

        label_embedding1->data = std::make_shared<int>(label1);
        label_embedding2->data = std::make_shared<int>(label2);
        input->output = frames_cat.at(time)->output;

        autodiff::eval(topo_order, autodiff::eval_funcs);

        return autodiff::get_output<double>(score);
    }

    void left_boundary_order2_score::accumulate_grad(double g, ilat::pair_fst const& f,
        std::tuple<int, int> e) const
    {
        int time = std::min<int>(std::max<int>(0, f.time(f.tail(e))), frames_cat.size() - 1);

        int label1 = f.output(e);
        int label2 = std::get<1>(f.tail(e));

        label_embedding1->data = std::make_shared<int>(label1);
        label_embedding2->data = std::make_shared<int>(label2);
        input->output = frames_cat.at(time)->output;

        autodiff::eval(topo_order, autodiff::eval_funcs);

        autodiff::clear_grad(topo_order);

        score->grad = std::make_shared<double>(g);

        autodiff::grad(score, autodiff::grad_funcs);
    }

    length_score::length_score(std::shared_ptr<autodiff::op_t> param)
        : param(param)
    {}

    double length_score::operator()(ilat::fst const& f,
        int e) const
    {
        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        auto& m = autodiff::get_output<la::matrix<double>>(param);

        return m(ell, std::min<int>(head_time - tail_time - 1, m.cols() - 1));
    }

    void length_score::accumulate_grad(double g, ilat::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::matrix<double>>(param);

        if (param->grad == nullptr) {
            la::matrix<double> m_grad;
            m_grad.resize(m.rows(), m.cols());
            param->grad = std::make_shared<la::matrix<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::matrix<double>>(param);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        m_grad(ell, std::min<int>(head_time - tail_time - 1, m.cols() - 1)) += g;
    }

    log_length_score::log_length_score(std::shared_ptr<autodiff::op_t> param)
        : param(param)
    {}

    double log_length_score::operator()(ilat::fst const& f,
        int e) const
    {
        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        auto& m = autodiff::get_output<la::matrix<double>>(param);
        double logd = std::log(head_time - tail_time);

        return m(ell, 0) * logd + m(ell, 1) * logd * logd;
    }

    void log_length_score::accumulate_grad(double g, ilat::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::matrix<double>>(param);

        if (param->grad == nullptr) {
            la::matrix<double> m_grad;
            m_grad.resize(m.rows(), m.cols());
            param->grad = std::make_shared<la::matrix<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::matrix<double>>(param);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        double logd = std::log(head_time - tail_time);

        m_grad(ell, 0) += g * logd;
        m_grad(ell, 1) += g * logd * logd;
    }

    bias0_score::bias0_score(std::shared_ptr<autodiff::op_t> param)
        : param(param)
    {}

    double bias0_score::operator()(ilat::fst const& f,
        int e) const
    {
        auto& v = autodiff::get_output<la::vector<double>>(param);

        return v(0);
    }

    void bias0_score::accumulate_grad(double g, ilat::fst const& f,
        int e) const
    {
        auto& v = autodiff::get_output<la::vector<double>>(param);

        if (param->grad == nullptr) {
            la::vector<double> v_grad;
            v_grad.resize(v.size());
            param->grad = std::make_shared<la::vector<double>>(std::move(v_grad));
        }

        auto& v_grad = autodiff::get_grad<la::vector<double>>(param);

        v_grad(0) += g;
    }

    bias1_score::bias1_score(std::shared_ptr<autodiff::op_t> param)
        : param(param)
    {}

    double bias1_score::operator()(ilat::fst const& f,
        int e) const
    {
        int ell = f.output(e) - 1;

        auto& v = autodiff::get_output<la::vector<double>>(param);

        return v(ell);
    }

    void bias1_score::accumulate_grad(double g, ilat::fst const& f,
        int e) const
    {
        auto& v = autodiff::get_output<la::vector<double>>(param);

        if (param->grad == nullptr) {
            la::vector<double> v_grad;
            v_grad.resize(v.size());
            param->grad = std::make_shared<la::vector<double>>(std::move(v_grad));
        }

        auto& v_grad = autodiff::get_grad<la::vector<double>>(param);

        int ell = f.output(e) - 1;

        v_grad(ell) += g;
    }

    external_score_order0::external_score_order0(std::shared_ptr<autodiff::op_t> param,
            std::vector<int> indices)
        : param(param), indices(indices)
    {}

    double external_score_order0::operator()(ilat::fst const& f,
        int e) const
    {
        auto& feat = f.data->feats.at(e);

        la::vector<double>& v = autodiff::get_output<la::vector<double>>(param);

        assert(indices.size() == v.size());

        double sum = 0;
        for (int i = 0; i < indices.size(); ++i) {
            sum += v(i) * feat.at(indices.at(i));
        }

        return sum;
    }

    void external_score_order0::accumulate_grad(double g, ilat::fst const& f,
        int e) const
    {
        auto& feat = f.data->feats.at(e);

        if (param->grad == nullptr) {
            la::vector<double>& v = autodiff::get_output<la::vector<double>>(param);

            la::vector<double> g_v;
            g_v.resize(v.size());
            param->grad = std::make_shared<la::vector<double>>(g_v);
        }

        la::vector<double>& g_v = autodiff::get_grad<la::vector<double>>(param);

        assert(indices.size() == g_v.size());

        for (int i = 0; i < indices.size(); ++i) {
            g_v(i) += g * feat.at(indices.at(i));
        }
    }

    external_score_order1::external_score_order1(std::shared_ptr<autodiff::op_t> param,
            std::vector<int> indices)
        : param(param), indices(indices)
    {}

    double external_score_order1::operator()(ilat::fst const& f,
        int e) const
    {
        auto& feat = f.data->feats.at(e);

        la::matrix<double>& m = autodiff::get_output<la::matrix<double>>(param);

        int ell = f.output(e) - 1;

        assert(indices.size() == m.cols());
        assert(ell < m.rows());

        double sum = 0;
        for (int i = 0; i < indices.size(); ++i) {
            sum += m(ell, i) * feat.at(indices.at(i));
        }

        return sum;
    }

    void external_score_order1::accumulate_grad(double g, ilat::fst const& f,
        int e) const
    {
        auto& feat = f.data->feats.at(e);

        if (param->grad == nullptr) {
            la::matrix<double>& m = autodiff::get_output<la::matrix<double>>(param);

            la::matrix<double> g_mat;
            g_mat.resize(m.rows(), m.cols());
            param->grad = std::make_shared<la::matrix<double>>(g_mat);
        }

        la::matrix<double>& g_mat = autodiff::get_grad<la::matrix<double>>(param);

        int ell = f.output(e) - 1;

        assert(indices.size() == g_mat.cols());
        assert(ell < g_mat.rows());

        for (int i = 0; i < indices.size(); ++i) {
            g_mat(ell, i) += g * feat.at(indices.at(i));
        }
    }

    edge_weight::edge_weight(std::shared_ptr<autodiff::op_t> param)
        : param(param)
    {}

    double edge_weight::operator()(ilat::fst const& f,
        int e) const
    {
        auto& v = autodiff::get_output<la::vector<double>>(param);

        return v(0) * f.weight(e);
    }

    void edge_weight::accumulate_grad(double g, ilat::fst const& f,
        int e) const
    {
        auto& v = autodiff::get_output<la::vector<double>>(param);

        if (param->grad == nullptr) {
            la::vector<double> v_grad;
            v_grad.resize(v.size());
            param->grad = std::make_shared<la::vector<double>>(std::move(v_grad));
        }

        auto& v_grad = autodiff::get_grad<la::vector<double>>(param);

        v_grad(0) += g * f.weight(e);
    }

    std::shared_ptr<tensor_tree::vertex> make_lstm_tensor_tree(
        int outer_layer, int inner_layer)
    {
        if (inner_layer == -1) {
            lstm::stacked_bi_lstm_tensor_tree_factory fac { outer_layer,
                std::make_shared<lstm::bi_lstm_tensor_tree_factory>(
                      lstm::bi_lstm_tensor_tree_factory {
                          std::make_shared<lstm::dyer_lstm_tensor_tree_factory>(
                              lstm::dyer_lstm_tensor_tree_factory{})
                          // std::make_shared<lstm::lstm_tensor_tree_factory>(
                          //     lstm::lstm_tensor_tree_factory{})
                      }
                )};

            return fac();
        } else {
            lstm::stacked_bi_lstm_tensor_tree_factory fac { outer_layer,
                std::make_shared<lstm::bi_lstm_tensor_tree_factory>(
                lstm::bi_lstm_tensor_tree_factory {
                    std::make_shared<lstm::multilayer_lstm_tensor_tree_factory>(
                    lstm::multilayer_lstm_tensor_tree_factory {
                       std::make_shared<lstm::dyer_lstm_tensor_tree_factory>(
                       lstm::dyer_lstm_tensor_tree_factory{}),
                       inner_layer
                    })
                })
            };

            return fac();
        }
    }

    std::tuple<int, int, std::shared_ptr<tensor_tree::vertex>, std::shared_ptr<tensor_tree::vertex>>
    load_lstm_param(std::string filename)
    {
        std::ifstream ifs { filename };
        std::string line;

        std::getline(ifs, line);
        auto parts = ebt::split(line);

        if (parts.size() == 1) {
            int layer = std::stoi(line);
            std::shared_ptr<tensor_tree::vertex> nn_param = make_lstm_tensor_tree(layer, -1);
            tensor_tree::load_tensor(nn_param, ifs);
            std::shared_ptr<tensor_tree::vertex> pred_param = nn::make_pred_tensor_tree();
            tensor_tree::load_tensor(pred_param, ifs);

            return std::make_tuple(layer, -1, nn_param, pred_param);
        } else if (parts.size() == 2) {
            int outer_layer = std::stoi(parts[0]);
            int inner_layer = std::stoi(parts[1]);
            std::shared_ptr<tensor_tree::vertex> nn_param = make_lstm_tensor_tree(outer_layer, inner_layer);
            tensor_tree::load_tensor(nn_param, ifs);
            std::shared_ptr<tensor_tree::vertex> pred_param = nn::make_pred_tensor_tree();
            tensor_tree::load_tensor(pred_param, ifs);

            return std::make_tuple(outer_layer, inner_layer, nn_param, pred_param);
        }
    }

    void save_lstm_param(
        std::shared_ptr<tensor_tree::vertex> nn_param,
        std::shared_ptr<tensor_tree::vertex> pred_param,
        std::string filename)
    {
        std::ofstream ofs { filename };

        ofs << nn_param->children.size() << std::endl;
        tensor_tree::save_tensor(nn_param, ofs);
        tensor_tree::save_tensor(pred_param, ofs);
    }

    void save_lstm_param(int outer_layer, int inner_layer,
        std::shared_ptr<tensor_tree::vertex> nn_param,
        std::shared_ptr<tensor_tree::vertex> pred_param,
        std::string filename)
    {
        std::ofstream ofs { filename };

        if (inner_layer == -1) {
            ofs << nn_param->children.size() << std::endl;
        } else {
            ofs << outer_layer << " " << inner_layer << std::endl;
        }
        tensor_tree::save_tensor(nn_param, ofs);
        tensor_tree::save_tensor(pred_param, ofs);
    }

    std::shared_ptr<lstm::transcriber>
    make_transcriber(
        int outer_layer,
        int inner_layer,
        std::unordered_map<std::string, std::string> const& args,
        std::default_random_engine *gen)
    {
        std::shared_ptr<lstm::lstm_step_transcriber> step;

        if (ebt::in(std::string("dropout"), args)) {
            step = std::make_shared<lstm::lstm_input_dropout_transcriber>(
                lstm::lstm_input_dropout_transcriber {
                    *gen, std::stod(args.at("dropout")),
                    std::make_shared<lstm::dyer_lstm_step_transcriber>(
                    lstm::dyer_lstm_step_transcriber{})
                });
        } else {
            step = std::make_shared<lstm::dyer_lstm_step_transcriber>(
                lstm::dyer_lstm_step_transcriber{});
        }

        lstm::layered_transcriber result;

        if (inner_layer == -1) {
            for (int i = 0; i < outer_layer; ++i) {
                std::shared_ptr<lstm::transcriber> trans = std::make_shared<lstm::bi_transcriber>(
                    lstm::bi_transcriber {
                        std::make_shared<lstm::lstm_transcriber>(
                        lstm::lstm_transcriber { step })
                    });

                if (i != outer_layer - 1) {
                    if (ebt::in(std::string("subsampling"), args)) {
                        trans = std::make_shared<lstm::subsampled_transcriber>(
                            lstm::subsampled_transcriber { 2, 0, trans });
                    }
                }

                result.layer.push_back(trans);
            }
        } else {
            for (int i = 0; i < outer_layer; ++i) {
                lstm::layered_transcriber layered_lstm;

                for (int j = 0; j < inner_layer; ++j) {
                    if (j != inner_layer - 1) {
                        layered_lstm.layer.push_back(
                            std::make_shared<lstm::lstm_transcriber>(
                            lstm::lstm_transcriber { step }));
                    } else {
                        layered_lstm.layer.push_back(
                            std::make_shared<lstm::lstm_transcriber>(
                            lstm::lstm_transcriber {
                                std::make_shared<lstm::lstm_output_dropout_transcriber>(
                                lstm::lstm_output_dropout_transcriber {
                                    *gen, std::stod(args.at("dropout")), step })
                            }));
                    }
                }

                std::shared_ptr<lstm::transcriber> trans = std::make_shared<lstm::bi_transcriber>(
                    lstm::bi_transcriber { std::make_shared<lstm::layered_transcriber>(layered_lstm) });

                if (i != outer_layer - 1) {
                    if (ebt::in(std::string("subsampling"), args)) {
                        trans = std::make_shared<lstm::subsampled_transcriber>(
                            lstm::subsampled_transcriber { 2, 0, trans });
                    }
                }

                result.layer.push_back(trans);
            }
        }

        return std::make_shared<lstm::layered_transcriber>(result);
    }

    std::shared_ptr<lstm::transcriber>
    make_transcriber(inference_args& i_args)
    {
        return make_transcriber(i_args.outer_layer, i_args.inner_layer,
            i_args.args, &i_args.gen);
    }

    void parse_inference_args(inference_args& i_args,
        std::unordered_map<std::string, std::string> const& args)
    {
        i_args.args = args;

        if (ebt::in(std::string("nn-param"), args)) {
            std::tie(i_args.outer_layer, i_args.inner_layer, i_args.nn_param, i_args.pred_param)
                = load_lstm_param(args.at("nn-param"));
        }

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

        if (ebt::in(std::string("features"), args)) {
            i_args.features = ebt::split(args.at("features"), ",");
        }

        if (ebt::in(std::string("param"), args)) {
            i_args.param = make_tensor_tree(i_args.features);

            tensor_tree::load_tensor(i_args.param, args.at("param"));
        }

        i_args.label_id = util::load_label_id(args.at("label"));

        i_args.id_label.resize(i_args.label_id.size());
        for (auto& p: i_args.label_id) {
            i_args.labels.push_back(p.second);
            i_args.id_label[p.second] = p.first;
        }

        if (ebt::in(std::string("seed"), args)) {
           i_args.gen = std::default_random_engine { std::stoul(args.at("seed")) };
        }
    }

    sample::sample(inference_args const& i_args)
    {
        graph_data.param = i_args.param;
    }

    void make_graph(sample& s, inference_args& i_args, int frames)
    {
        if (ebt::in(std::string("edge-drop"), i_args.args)) {
            s.graph_data.fst = make_random_graph(frames,
                i_args.label_id, i_args.id_label, i_args.min_seg, i_args.max_seg, i_args.stride,
                std::stod(i_args.args.at("edge-drop")), i_args.gen);
            s.graph_data.topo_order = std::make_shared<std::vector<int>>(
                ::fst::topo_order(*s.graph_data.fst));
        } else {
            s.graph_data.fst = make_graph(frames,
                i_args.label_id, i_args.id_label, i_args.min_seg, i_args.max_seg, i_args.stride);
            s.graph_data.topo_order = std::make_shared<std::vector<int>>(
                ::fst::topo_order(*s.graph_data.fst));
        }
    }

    void make_graph(sample& s, inference_args& i_args)
    {
        make_graph(s, i_args, s.frames.size());
    }

    void parse_learning_args(learning_args& l_args,
        std::unordered_map<std::string, std::string> const& args)
    {
        parse_inference_args(l_args, args);

        if (ebt::in(std::string("opt-data"), args)) {
            l_args.opt_data = make_tensor_tree(l_args.features);
            tensor_tree::load_tensor(l_args.opt_data, args.at("opt-data"));
        }

        if (ebt::in(std::string("nn-opt-data"), args)) {
            std::tie(l_args.outer_layer, l_args.inner_layer, l_args.nn_opt_data, l_args.pred_opt_data)
                = load_lstm_param(args.at("nn-opt-data"));
        }

        l_args.l2 = 0;
        if (ebt::in(std::string("l2"), args)) {
            l_args.l2 = std::stod(args.at("l2"));
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

        if (ebt::in(std::string("adam-beta1"), args)) {
            l_args.adam_beta1 = std::stod(args.at("adam-beta1"));
        }

        if (ebt::in(std::string("adam-beta2"), args)) {
            l_args.adam_beta2 = std::stod(args.at("adam-beta2"));
        }
    }

    learning_sample::learning_sample(learning_args const& l_args)
        : sample(l_args)
    {
        gold_data.param = l_args.param;
    }

    loss_func::~loss_func()
    {}

    hinge_loss::hinge_loss(fscrf_data& graph_data,
            std::vector<segcost::segment<int>> const& gt_segs,
            std::vector<int> const& sils,
            double cost_scale)
        : graph_data(graph_data), sils(sils), cost_scale(cost_scale)
    {
        auto old_weight_func = graph_data.weight_func;
        graph_data.weight_func = std::make_shared<scrf::mul<ilat::fst>>(
            scrf::mul<ilat::fst>(std::make_shared<scrf::seg_cost<ilat::fst>>(
                scrf::make_overlap_cost<ilat::fst>(gt_segs, sils)), -1));
        gold_path_data.fst = scrf::shortest_path(graph_data);
        graph_data.weight_func = old_weight_func;
        gold_path_data.weight_func = graph_data.weight_func;

        fscrf_fst gold_path { gold_path_data };

        for (auto& e: gold_path.edges()) {
            int tail_time = gold_path.time(gold_path.tail(e));
            int head_time = gold_path.time(gold_path.head(e));

            gold_segs.push_back(segcost::segment<int> { tail_time, head_time, gold_path.output(e) });
        }

        graph_data.cost_func = std::make_shared<scrf::mul<ilat::fst>>(
            scrf::mul<ilat::fst>(std::make_shared<scrf::seg_cost<ilat::fst>>(
                scrf::make_overlap_cost<ilat::fst>(gold_segs, sils)), cost_scale));

        gold_path_data.cost_func = graph_data.cost_func;

        auto& id_symbol = *graph_data.fst->data->id_symbol;

        double gold_cost = 0;
        double gold_score = 0;
        std::cout << "gold:";
        for (auto& e: gold_path.edges()) {
            double c = (*gold_path_data.cost_func)(*gold_path_data.fst, e);
            gold_cost += c;
            gold_score += gold_path.weight(e);

            std::cout << " " << id_symbol[gold_path.output(e)] << " (" << c << ")";
        }
        std::cout << std::endl;
        std::cout << "gold cost: " << gold_cost << std::endl;
        std::cout << "gold score: " << gold_score << std::endl;

        scrf::composite_weight<ilat::fst> weight_cost;
        weight_cost.weights.push_back(graph_data.cost_func);
        weight_cost.weights.push_back(graph_data.weight_func);
        graph_data.weight_func = std::make_shared<scrf::composite_weight<ilat::fst>>(weight_cost);
        graph_path_data.fst = scrf::shortest_path(graph_data);
        graph_path_data.weight_func = graph_data.weight_func;

        fscrf_fst graph_path { graph_path_data };

        double cost_aug_cost = 0;
        double cost_aug_score = 0;
        std::cout << "cost aug:";
        for (auto& e: graph_path.edges()) {
            cost_aug_cost += (*graph_data.cost_func)(*graph_path_data.fst, e);
            cost_aug_score += graph_path.weight(e);

            std::cout << " " << id_symbol[graph_path.output(e)];
        }
        std::cout << std::endl;
        std::cout << "cost aug cost: " << cost_aug_cost << std::endl;
        std::cout << "cost aug score: " << cost_aug_score << std::endl;
    }

    double hinge_loss::loss() const
    {
        double result = 0;

        fscrf_fst gold_path { gold_path_data };

        for (auto& e: gold_path.edges()) {
            result -= gold_path.weight(e);
        }

        fscrf_fst graph_path { graph_path_data };

        for (auto& e: graph_path.edges()) {
            result += graph_path.weight(e);
        }

        return result;
    }

    void hinge_loss::grad() const
    {
        fscrf_fst gold_path { gold_path_data };

        for (auto& e: gold_path.edges()) {
            gold_path_data.weight_func->accumulate_grad(-1, *gold_path_data.fst, e);
        }

        fscrf_fst graph_path { graph_path_data };

        for (auto& e: graph_path.edges()) {
            graph_path_data.weight_func->accumulate_grad(1, *graph_path_data.fst, e);
        }
    }

    hinge_loss_gt::hinge_loss_gt(fscrf_data& graph_data,
            std::vector<segcost::segment<int>> const& gt_segs,
            std::vector<int> const& sils,
            double cost_scale)
        : graph_data(graph_data), sils(sils), cost_scale(cost_scale)
    {
        gold_path_data.weight_func = graph_data.weight_func;

        ilat::fst_data gold_data;
        gold_data.symbol_id = graph_data.fst->data->symbol_id;
        gold_data.id_symbol = graph_data.fst->data->id_symbol;

        int v = 0;
        int e = 0;

        for (int i = 0; i < gt_segs.size(); ++i) {
            ilat::add_vertex(gold_data, v, ilat::vertex_data { gt_segs[i].start_time });
            ++v;

            ilat::add_vertex(gold_data, v, ilat::vertex_data { gt_segs[i].end_time });
            ++v;

            ilat::add_edge(gold_data, e, ilat::edge_data { v - 2, v - 1, 0, gt_segs[i].label, gt_segs[i].label });
            ++e;
        }

        ilat::fst gold_fst;
        gold_fst.data = std::make_shared<ilat::fst_data>(gold_data);

        gold_path_data.fst = std::make_shared<ilat::fst>(gold_fst);

        graph_data.cost_func = std::make_shared<scrf::mul<ilat::fst>>(
            scrf::mul<ilat::fst>(std::make_shared<scrf::seg_cost<ilat::fst>>(
                scrf::make_overlap_cost<ilat::fst>(gt_segs, sils)), cost_scale));

        gold_path_data.cost_func = graph_data.cost_func;

        fscrf_fst gold_path { gold_path_data };

        auto& id_symbol = *graph_data.fst->data->id_symbol;

        double gold_cost = 0;
        double gold_score = 0;
        std::cout << "gold:";
        for (auto& e: gold_path.edges()) {
            double c = (*gold_path_data.cost_func)(*gold_path_data.fst, e);
            gold_cost += c;
            gold_score += gold_path.weight(e);

            std::cout << " " << id_symbol[gold_path.output(e)] << " (" << c << ")";
        }
        std::cout << std::endl;
        std::cout << "gold cost: " << gold_cost << std::endl;
        std::cout << "gold score: " << gold_score << std::endl;

        scrf::composite_weight<ilat::fst> weight_cost;
        weight_cost.weights.push_back(graph_data.cost_func);
        weight_cost.weights.push_back(graph_data.weight_func);
        graph_data.weight_func = std::make_shared<scrf::composite_weight<ilat::fst>>(weight_cost);
        graph_path_data.fst = scrf::shortest_path(graph_data);
        graph_path_data.weight_func = graph_data.weight_func;

        fscrf_fst graph_path { graph_path_data };

        double cost_aug_cost = 0;
        double cost_aug_score = 0;
        std::cout << "cost aug:";
        for (auto& e: graph_path.edges()) {
            cost_aug_cost += (*graph_data.cost_func)(*graph_path_data.fst, e);
            cost_aug_score += graph_path.weight(e);

            std::cout << " " << id_symbol[graph_path.output(e)];
        }
        std::cout << std::endl;
        std::cout << "cost aug cost: " << cost_aug_cost << std::endl;
        std::cout << "cost aug score: " << cost_aug_score << std::endl;
    }

    double hinge_loss_gt::loss() const
    {
        double result = 0;

        fscrf_fst gold_path { gold_path_data };

        for (auto& e: gold_path.edges()) {
            result -= gold_path.weight(e);
        }

        fscrf_fst graph_path { graph_path_data };

        for (auto& e: graph_path.edges()) {
            result += graph_path.weight(e);
        }

        return result;
    }

    void hinge_loss_gt::grad() const
    {
        fscrf_fst gold_path { gold_path_data };

        for (auto& e: gold_path.edges()) {
            gold_path_data.weight_func->accumulate_grad(-1, *gold_path_data.fst, e);
        }

        fscrf_fst graph_path { graph_path_data };

        for (auto& e: graph_path.edges()) {
            graph_path_data.weight_func->accumulate_grad(1, *graph_path_data.fst, e);
        }
    }

    log_loss::log_loss(fscrf_data& graph_data,
        std::vector<segcost::segment<int>> const& gt_segs,
        std::vector<int> const& sils)
        : graph_data(graph_data)
    {
        auto old_weight_func = graph_data.weight_func;
        graph_data.weight_func = std::make_shared<scrf::mul<ilat::fst>>(
            scrf::mul<ilat::fst>(std::make_shared<scrf::seg_cost<ilat::fst>>(
                scrf::make_overlap_cost<ilat::fst>(gt_segs, sils)), -1));
        gold_path_data.fst = scrf::shortest_path(graph_data);
        graph_data.weight_func = old_weight_func;
        gold_path_data.weight_func = graph_data.weight_func;

        fscrf_fst gold_path { gold_path_data };

        for (auto& e: gold_path.edges()) {
            int tail_time = gold_path.time(gold_path.tail(e));
            int head_time = gold_path.time(gold_path.head(e));

            gold_segs.push_back(segcost::segment<int> { tail_time, head_time, gold_path.output(e) });
        }

        gold_path_data.cost_func = std::make_shared<scrf::mul<ilat::fst>>(
            scrf::mul<ilat::fst>(std::make_shared<scrf::seg_cost<ilat::fst>>(
                scrf::make_overlap_cost<ilat::fst>(gold_segs, sils)), 1));

        auto& id_symbol = *graph_data.fst->data->id_symbol;

        double gold_cost = 0;
        double gold_score = 0;
        std::cout << "gold:";
        for (auto& e: gold_path.edges()) {
            double c = (*gold_path_data.cost_func)(*gold_path_data.fst, e);
            gold_cost += c;
            gold_score += gold_path.weight(e);

            std::cout << " " << id_symbol[gold_path.output(e)] << " (" << c << ")";
        }
        std::cout << std::endl;
        std::cout << "gold cost: " << gold_cost << std::endl;
        std::cout << "gold score: " << gold_score << std::endl;

        fscrf_fst graph { graph_data };

        forward.merge(graph, *graph_data.topo_order);

        auto rev_topo_order = *graph_data.topo_order;
        std::reverse(rev_topo_order.begin(), rev_topo_order.end());

        backward.merge(graph, rev_topo_order);

        for (auto& f: graph.finals()) {
            std::cout << "forward: " << forward.extra[f] << std::endl;
        }

        for (auto& i: graph.initials()) {
            std::cout << "backward: " << backward.extra[i] << std::endl;
        }
    }

    double log_loss::loss() const
    {
        double result = 0;

        fscrf_fst gold_path { gold_path_data };

        for (auto& e: gold_path.edges()) {
            result -= gold_path.weight(e);
        }

        fscrf_fst graph { graph_data };

        result += forward.extra.at(graph.finals().front());

        return result;
    }

    void log_loss::grad() const
    {
        fscrf_fst gold_path { gold_path_data };

        for (auto& e: gold_path.edges()) {
            gold_path_data.weight_func->accumulate_grad(-1, *gold_path_data.fst, e);
        }

        fscrf_fst graph { graph_data };

        double logZ = forward.extra.at(graph.finals().front());

        for (auto& e: graph.edges()) {
            graph_data.weight_func->accumulate_grad(
                std::exp(forward.extra.at(graph.tail(e)) + graph.weight(e)
                    + backward.extra.at(graph.head(e)) - logZ), *graph_data.fst, e);
        }
    }

    ilat::fst make_label_fst(std::vector<int> const& label_seq,
        std::unordered_map<std::string, int> const& label_id,
        std::vector<std::string> const& id_label)
    {
        ilat::fst_data data;

        data.symbol_id = std::make_shared<std::unordered_map<std::string, int>>(label_id);
        data.id_symbol = std::make_shared<std::vector<std::string>>(id_label);

        int u = 0;
        ilat::add_vertex(data, u, ilat::vertex_data { u });

        for (int i = 0; i < label_seq.size(); ++i) {
            int v = data.vertices.size();
            ilat::add_vertex(data, v, ilat::vertex_data { v });

            int e = data.edges.size();
            ilat::add_edge(data, e, ilat::edge_data { u, v, 0,
                label_seq[i], label_seq[i] });

            u = v;
        }

        data.initials.push_back(0);
        data.finals.push_back(data.vertices.size() - 1);

        ilat::fst f;
        f.data = std::make_shared<ilat::fst_data>(data);

        return f;
    }

    marginal_log_loss::marginal_log_loss(fscrf_data& graph_data,
        std::vector<int> const& label_seq)
        : graph_data(graph_data)
    {
        fscrf_fst graph { graph_data };

        forward_graph.merge(graph, *graph_data.topo_order);

        auto rev_topo_order = *graph_data.topo_order;
        std::reverse(rev_topo_order.begin(), rev_topo_order.end());

        backward_graph.merge(graph, rev_topo_order);

        for (auto& f: graph.finals()) {
            std::cout << "forward: " << forward_graph.extra[f] << std::endl;
        }

        for (auto& i: graph.initials()) {
            std::cout << "backward: " << backward_graph.extra[i] << std::endl;
        }

        ilat::fst& graph_fst = *graph_data.fst;
        auto& label_id = *graph_fst.data->symbol_id;
        auto& id_label = *graph_fst.data->id_symbol;

        ilat::fst label_fst = make_label_fst(label_seq, label_id, id_label);

        ilat::lazy_pair_mode1 composed_fst { label_fst, graph_fst };

        pair_data.fst = std::make_shared<ilat::lazy_pair_mode1>(composed_fst);
        pair_data.weight_func = std::make_shared<mode2_weight>(
            mode2_weight { graph_data.weight_func });
        pair_data.topo_order = std::make_shared<std::vector<std::tuple<int, int>>>(
            fst::topo_order(composed_fst));

        fscrf_pair_fst pair { pair_data };

        forward_label.merge(pair, *pair_data.topo_order);

        std::vector<std::tuple<int, int>> rev_pair_topo_order = *pair_data.topo_order;
        std::reverse(rev_pair_topo_order.begin(), rev_pair_topo_order.end());
        backward_label.merge(pair, rev_pair_topo_order);

        for (auto& f: pair.finals()) {
            std::cout << "forward: " << forward_label.extra[f] << std::endl;
        }

        for (auto& i: pair.initials()) {
            std::cout << "backward: " << backward_label.extra[i] << std::endl;
        }

    }

    double marginal_log_loss::loss() const
    {
        double result = 0;

        fscrf_pair_fst pair { pair_data };

        result -= forward_label.extra.at(pair.finals().front());

        fscrf_fst graph { graph_data };

        result += forward_graph.extra.at(graph.finals().front());

        return result;
    }

    void marginal_log_loss::grad() const
    {
        fscrf_pair_fst pair { pair_data };

        double logZ1 = forward_label.extra.at(pair.finals().front());

        for (auto& e: pair.edges()) {
            if (!ebt::in(pair.tail(e), forward_label.extra) ||
                    !ebt::in(pair.head(e), backward_label.extra)) {
                continue;
            }

            pair_data.weight_func->accumulate_grad(
                -std::exp(forward_label.extra.at(pair.tail(e)) + pair.weight(e)
                    + backward_label.extra.at(pair.head(e)) - logZ1), *pair_data.fst, e);
        }

        fscrf_fst graph { graph_data };

        double logZ2 = forward_graph.extra.at(graph.finals().front());

        for (auto& e: graph.edges()) {
            graph_data.weight_func->accumulate_grad(
                std::exp(forward_graph.extra.at(graph.tail(e)) + graph.weight(e)
                    + backward_graph.extra.at(graph.head(e)) - logZ2), *graph_data.fst, e);
        }
    }

    latent_hinge_loss::latent_hinge_loss(fscrf_data& graph_data,
            std::vector<int> const& label_seq,
            std::vector<int> const& sils,
            double cost_scale)
        : graph_data(graph_data), sils(sils), cost_scale(cost_scale)
    {
        ilat::fst& graph_fst = *graph_data.fst;
        auto& label_id = *graph_fst.data->symbol_id;
        auto& id_label = *graph_fst.data->id_symbol;

        ilat::fst label_fst = make_label_fst(label_seq, label_id, id_label);

        ilat::lazy_pair_mode1 composed_fst { label_fst, graph_fst };

        pair_data.fst = std::make_shared<ilat::lazy_pair_mode1>(composed_fst);
        pair_data.weight_func = std::make_shared<mode2_weight>(
            mode2_weight { graph_data.weight_func });
        pair_data.topo_order = std::make_shared<std::vector<std::tuple<int, int>>>(
            fst::topo_order(composed_fst));

        fscrf_pair_fst pair { pair_data };

        gold_path_data.fst = scrf::shortest_path(pair_data);
        gold_path_data.weight_func = pair_data.weight_func;

        fscrf_fst graph { graph_data };
        fscrf_pair_fst gold_path { gold_path_data };

        for (auto& e: gold_path.edges()) {
            int tail_time = graph.time(std::get<1>(gold_path.tail(e)));
            int head_time = graph.time(std::get<1>(gold_path.head(e)));

            gold_segs.push_back(segcost::segment<int> { tail_time, head_time, gold_path.output(e) });
        }

        graph_data.cost_func = std::make_shared<scrf::mul<ilat::fst>>(
            scrf::mul<ilat::fst>(std::make_shared<scrf::seg_cost<ilat::fst>>(
                scrf::make_overlap_cost<ilat::fst>(gold_segs, sils)), cost_scale));

        double gold_score = 0;
        std::cout << "gold:";
        for (auto& e: gold_path.edges()) {
            gold_score += gold_path.weight(e);

            std::cout << " " << id_label[gold_path.output(e)]
                << " (" << graph.time(std::get<1>(gold_path.head(e))) << ")";
        }
        std::cout << std::endl;
        std::cout << "gold score: " << gold_score << std::endl;

        scrf::composite_weight<ilat::fst> weight_cost;
        weight_cost.weights.push_back(graph_data.cost_func);
        weight_cost.weights.push_back(graph_data.weight_func);
        graph_data.weight_func = std::make_shared<scrf::composite_weight<ilat::fst>>(weight_cost);
        graph_path_data.fst = scrf::shortest_path(graph_data);
        graph_path_data.weight_func = graph_data.weight_func;

        fscrf_fst graph_path { graph_path_data };

        double cost_aug_cost = 0;
        double cost_aug_score = 0;
        std::cout << "cost aug:";
        for (auto& e: graph_path.edges()) {
            cost_aug_cost += (*graph_data.cost_func)(*graph_path_data.fst, e);
            cost_aug_score += graph_path.weight(e);

            std::cout << " " << id_label[graph_path.output(e)];
        }
        std::cout << std::endl;
        std::cout << "cost aug cost: " << cost_aug_cost << std::endl;
        std::cout << "cost aug score: " << cost_aug_score << std::endl;
    }

    double latent_hinge_loss::loss() const
    {
        double gold_score = 0;

        fscrf_pair_fst gold_path { gold_path_data };

        for (auto& e: gold_path.edges()) {
            gold_score += gold_path.weight(e);
        }

        fscrf_fst graph_path { graph_path_data };

        double graph_score = 0;

        for (auto& e: graph_path.edges()) {
            graph_score += graph_path.weight(e);
        }

        std::cout << "debug loss: " << -gold_score + graph_score << std::endl;

        return -gold_score + graph_score;
    }

    void latent_hinge_loss::grad() const
    {
        fscrf_pair_fst gold_path { gold_path_data };

        for (auto& e: gold_path.edges()) {
            gold_path_data.weight_func->accumulate_grad(-1, *gold_path.data.fst, e);
        }

        fscrf_fst graph_path { graph_path_data };

        for (auto& e: graph_path.edges()) {
            graph_path_data.weight_func->accumulate_grad(1, *graph_path.data.fst, e);
        }
    }

    mode1_weight::mode1_weight(std::shared_ptr<scrf::scrf_weight<ilat::fst>> weight)
        : weight(weight)
    {}

    double mode1_weight::operator()(ilat::pair_fst const& fst,
        std::tuple<int, int> e) const
    {
        return (*weight)(fst.fst1(), std::get<0>(e));
    }

    void mode1_weight::accumulate_grad(double g, ilat::pair_fst const& fst,
        std::tuple<int, int> e) const
    {
        weight->accumulate_grad(g, fst.fst1(), std::get<0>(e));
    }

    void mode1_weight::grad() const
    {
        weight->grad();
    }

    mode2_weight::mode2_weight(std::shared_ptr<scrf::scrf_weight<ilat::fst>> weight)
        : weight(weight)
    {}

    double mode2_weight::operator()(ilat::pair_fst const& fst,
        std::tuple<int, int> e) const
    {
        return (*weight)(fst.fst2(), std::get<1>(e));
    }

    void mode2_weight::accumulate_grad(double g, ilat::pair_fst const& fst,
        std::tuple<int, int> e) const
    {
        weight->accumulate_grad(g, fst.fst2(), std::get<1>(e));
    }

    void mode2_weight::grad() const
    {
        weight->grad();
    }

    std::shared_ptr<scrf::scrf_weight<ilat::fst>> make_lat_weights(
        std::vector<std::string> const& features,
        std::shared_ptr<tensor_tree::vertex> var_tree)
    {
        scrf::composite_weight<ilat::fst> weight_func;
        int feat_idx = 0;
    
        for (auto& k: features) {
            if (ebt::startswith(k, "ext0")) {
                auto parts = ebt::split(k, ":");
                parts = ebt::split(parts[1], "+");
                std::vector<int> dims;
    
                for (auto& p: parts) {
                    std::vector<std::string> range = ebt::split(p, "-");
                    if (range.size() == 2) {
                        for (int i = std::stoi(range[0]); i <= std::stoi(range[1]); ++i) {
                            dims.push_back(i);
                        }
                    } else if (range.size() == 1) {
                        dims.push_back(std::stoi(p));
                    } else {
                        std::cerr << "unknown external feature format: " << k << std::endl;
                    }
                }
    
                weight_func.weights.push_back(std::make_shared<fscrf::external_score_order0>(
                    fscrf::external_score_order0 { tensor_tree::get_var(var_tree->children[feat_idx]), dims }));
    
                ++feat_idx;
            } else if (ebt::startswith(k, "ext1")) {
                auto parts = ebt::split(k, ":");
                parts = ebt::split(parts[1], "+");
                std::vector<int> dims;
    
                for (auto& p: parts) {
                    std::vector<std::string> range = ebt::split(p, "-");
                    if (range.size() == 2) {
                        for (int i = std::stoi(range[0]); i <= std::stoi(range[1]); ++i) {
                            dims.push_back(i);
                        }
                    } else if (range.size() == 1) {
                        dims.push_back(std::stoi(p));
                    } else {
                        std::cerr << "unknown external feature format: " << k << std::endl;
                    }
                }
    
                weight_func.weights.push_back(std::make_shared<fscrf::external_score_order1>(
                    fscrf::external_score_order1 { tensor_tree::get_var(var_tree->children[feat_idx]), dims }));
    
                ++feat_idx;
            } else if (k == "bias0") {
                weight_func.weights.push_back(std::make_shared<fscrf::bias0_score>(
                    fscrf::bias0_score { tensor_tree::get_var(var_tree->children[feat_idx]) }));
    
                ++feat_idx;
            } else if (k == "bias1") {
                weight_func.weights.push_back(std::make_shared<fscrf::bias1_score>(
                    fscrf::bias1_score { tensor_tree::get_var(var_tree->children[feat_idx]) }));
    
                ++feat_idx;
            } else {
                std::cout << "unknown feature: " << k << std::endl;
                exit(1);
            }
        }
    
        return std::make_shared<scrf::composite_weight<ilat::fst>>(weight_func);
    }

    std::shared_ptr<scrf::scrf_weight<ilat::pair_fst>> make_pair_weights(
        std::vector<std::string> const& features,
        std::shared_ptr<tensor_tree::vertex> var_tree,
        std::vector<std::vector<double>> const& frames)
    {
        scrf::composite_weight<ilat::pair_fst> weight_func;
        int feat_idx = 0;
    
        for (auto& k: features) {
            if (k == "lm") {
                weight_func.weights.push_back(std::make_shared<mode2_weight>(mode2_weight {
                    std::make_shared<edge_weight>(edge_weight { tensor_tree::get_var(var_tree->children[feat_idx]) })
                }));

                ++feat_idx;
            } else if (k == "lat-weight") {
                weight_func.weights.push_back(std::make_shared<mode1_weight>(mode1_weight {
                    std::make_shared<edge_weight>(edge_weight { tensor_tree::get_var(var_tree->children[feat_idx]) })
                }));

                ++feat_idx;
            } else if (k == "boundary2") {
                weight_func.weights.push_back(std::make_shared<left_boundary_order2_score>(left_boundary_order2_score {
                    var_tree->children[feat_idx],
                    frames,
                    3
                }));

                ++feat_idx;
            } else if (k == "bias0") {
                weight_func.weights.push_back(std::make_shared<mode1_weight>(mode1_weight {
                    std::make_shared<fscrf::bias0_score>(fscrf::bias0_score {
                        tensor_tree::get_var(var_tree->children[feat_idx])
                    })
                }));
    
                ++feat_idx;
            } else {
                std::cout << "unknown feature: " << k << std::endl;
                exit(1);
            }
        }
    
        return std::make_shared<scrf::composite_weight<ilat::pair_fst>>(weight_func);
    }

    hinge_loss_pair::hinge_loss_pair(fscrf_pair_data& graph_data,
            std::vector<segcost::segment<int>> const& gt_segs,
            std::vector<int> const& sils,
            double cost_scale)
        : graph_data(graph_data), sils(sils), cost_scale(cost_scale)
    {
        auto old_weight_func = graph_data.weight_func;

        graph_data.weight_func = std::make_shared<mode1_weight>(
            mode1_weight { std::make_shared<scrf::mul<ilat::fst>>(
            scrf::mul<ilat::fst>(std::make_shared<scrf::seg_cost<ilat::fst>>(
                scrf::make_overlap_cost<ilat::fst>(gt_segs, sils)), -1)) });

        gold_path_data.fst = scrf::shortest_path(graph_data);
        graph_data.weight_func = old_weight_func;
        gold_path_data.weight_func = graph_data.weight_func;

        auto& id_symbol = *graph_data.fst->fst1().data->id_symbol;

        fscrf_pair_fst gold_path { gold_path_data };

        for (auto& e: gold_path.edges()) {
            int tail_time = gold_path.time(gold_path.tail(e));
            int head_time = gold_path.time(gold_path.head(e));

            gold_segs.push_back(segcost::segment<int> { tail_time, head_time, gold_path.output(e) });
        }

        std::cout << std::endl;

        graph_data.cost_func = std::make_shared<mode1_weight>(
            mode1_weight { std::make_shared<scrf::mul<ilat::fst>>(
            scrf::mul<ilat::fst>(std::make_shared<scrf::seg_cost<ilat::fst>>(
                scrf::make_overlap_cost<ilat::fst>(gold_segs, sils)), cost_scale)) });

        gold_path_data.cost_func = graph_data.cost_func;

        double gold_cost = 0;
        double gold_score = 0;
        std::cout << "gold:";
        for (auto& e: gold_path.edges()) {
            double c = (*gold_path_data.cost_func)(*gold_path_data.fst, e);
            gold_cost += c;
            gold_score += gold_path.weight(e);

            std::cout << " " << id_symbol[gold_path.output(e)] << " (" << c << ")";
        }
        std::cout << std::endl;
        std::cout << "gold cost: " << gold_cost << std::endl;
        std::cout << "gold score: " << gold_score << std::endl;

        scrf::composite_weight<ilat::pair_fst> weight_cost;
        weight_cost.weights.push_back(graph_data.cost_func);
        weight_cost.weights.push_back(graph_data.weight_func);
        graph_data.weight_func = std::make_shared<scrf::composite_weight<ilat::pair_fst>>(weight_cost);
        graph_path_data.fst = scrf::shortest_path(graph_data);
        graph_path_data.weight_func = graph_data.weight_func;

        fscrf_pair_fst graph_path { graph_path_data };

        double cost_aug_cost = 0;
        double cost_aug_score = 0;
        std::cout << "cost aug:";
        for (auto& e: graph_path.edges()) {
            cost_aug_cost += (*graph_data.cost_func)(*graph_path_data.fst, e);
            cost_aug_score += graph_path.weight(e);

            std::cout << " " << id_symbol[graph_path.output(e)];
        }
        std::cout << std::endl;
        std::cout << "cost aug cost: " << cost_aug_cost << std::endl;
        std::cout << "cost aug score: " << cost_aug_score << std::endl;
    }

    double hinge_loss_pair::loss() const
    {
        double result = 0;

        fscrf_pair_fst gold_path { gold_path_data };

        for (auto& e: gold_path.edges()) {
            result -= gold_path.weight(e);
        }

        fscrf_pair_fst graph_path { graph_path_data };

        for (auto& e: graph_path.edges()) {
            result += graph_path.weight(e);
        }

        return result;
    }

    void hinge_loss_pair::grad() const
    {
        fscrf_pair_fst gold_path { gold_path_data };

        for (auto& e: gold_path.edges()) {
            gold_path_data.weight_func->accumulate_grad(-1, *gold_path_data.fst, e);
        }

        fscrf_pair_fst graph_path { graph_path_data };

        for (auto& e: graph_path.edges()) {
            graph_path_data.weight_func->accumulate_grad(1, *graph_path_data.fst, e);
        }
    }

}
