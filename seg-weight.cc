#include "seg/seg-weight.h"

namespace seg {

    mode1_weight::mode1_weight(std::shared_ptr<seg_weight<ifst::fst>> weight)
        : weight(weight)
    {}

    double mode1_weight::operator()(fst::pair_fst<ifst::fst, ifst::fst> const& fst,
        std::tuple<int, int> e) const
    {
        return (*weight)(fst.fst1(), std::get<0>(e));
    }

    void mode1_weight::accumulate_grad(double g, fst::pair_fst<ifst::fst, ifst::fst> const& fst,
        std::tuple<int, int> e) const
    {
        weight->accumulate_grad(g, fst.fst1(), std::get<0>(e));
    }

    void mode1_weight::grad() const
    {
        weight->grad();
    }

    mode2_weight::mode2_weight(std::shared_ptr<seg_weight<ifst::fst>> weight)
        : weight(weight)
    {}

    double mode2_weight::operator()(fst::pair_fst<ifst::fst, ifst::fst> const& fst,
        std::tuple<int, int> e) const
    {
        return (*weight)(fst.fst2(), std::get<1>(e));
    }

    void mode2_weight::accumulate_grad(double g, fst::pair_fst<ifst::fst, ifst::fst> const& fst,
        std::tuple<int, int> e) const
    {
        weight->accumulate_grad(g, fst.fst2(), std::get<1>(e));
    }

    void mode2_weight::grad() const
    {
        weight->grad();
    }

    frame_sum_score::frame_sum_score(std::shared_ptr<autodiff::op_t> frames)
        : frames(frames)
    {}

    double frame_sum_score::operator()(ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::cpu::tensor_like<double>>(frames);

        double sum = 0;

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        for (int t = tail_time; t < head_time; ++t) {
            sum += m({t, ell});
        }

        return sum;
    }

    void frame_sum_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::cpu::tensor_like<double>>(frames);

        if (frames->grad == nullptr) {
            la::cpu::tensor<double> m_grad;
            la::cpu::resize_as(m_grad, m);
            frames->grad = std::make_shared<la::cpu::tensor<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::cpu::tensor_like<double>>(frames);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        for (int t = tail_time; t < head_time; ++t) {
            m_grad({t, ell}) += g;
        }
    }

    frame_avg_score::frame_avg_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames)
        : param(param), frames(frames)
    {
        score = autodiff::rtmul(param, frames);
    }

    double frame_avg_score::operator()(ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::cpu::tensor_like<double>>(score);

        double sum = 0;

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        for (int t = tail_time; t < head_time; ++t) {
            sum += m({ell, t});
        }

        return (head_time <= tail_time ? 0 : sum / (head_time - tail_time));
    }

    void frame_avg_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::cpu::tensor_like<double>>(score);

        if (score->grad == nullptr) {
            la::cpu::tensor<double> m_grad;
            la::cpu::resize_as(m_grad, m);
            score->grad = std::make_shared<la::cpu::tensor<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::cpu::tensor_like<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        for (int t = tail_time; t < head_time; ++t) {
            m_grad({ell, t}) += g / (head_time - tail_time);
        }
    }

    void frame_avg_score::grad() const
    {
        autodiff::eval_vertex(score, autodiff::grad_funcs);
    }

    frame_samples_score::frame_samples_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames, double scale)
        : param(param), frames(frames), scale(scale)
    {
        score = autodiff::rtmul(param, frames);
    }

    double frame_samples_score::operator()(ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::cpu::tensor<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        int dur = head_time - tail_time;

        return m({ell, int(tail_time + dur * scale)});
    }

    void frame_samples_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::cpu::tensor<double>>(score);

        if (score->grad == nullptr) {
            la::cpu::tensor<double> m_grad;
            la::cpu::resize_as(m_grad, m);
            score->grad = std::make_shared<la::cpu::tensor<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::cpu::tensor<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        int dur = head_time - tail_time;

        m_grad({ell, int(tail_time + dur * scale)}) += g;
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
    }

    double left_boundary_score::operator()(ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::cpu::tensor<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        return m({ell, std::max<int>(tail_time + shift, 0)});
    }

    void left_boundary_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::cpu::tensor<double>>(score);

        if (score->grad == nullptr) {
            la::cpu::tensor<double> m_grad;
            la::cpu::resize_as(m_grad, m);
            score->grad = std::make_shared<la::cpu::tensor<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::cpu::tensor<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        m_grad({ell, std::max<int>(tail_time + shift, 0)}) += g;
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
    }

    double right_boundary_score::operator()(ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::cpu::tensor<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        return m({ell, std::min<int>(head_time + shift, m.size(1) - 1)});
    }

    void right_boundary_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::cpu::tensor<double>>(score);

        if (score->grad == nullptr) {
            la::cpu::tensor<double> m_grad;
            la::cpu::resize_as(m_grad, m);
            score->grad = std::make_shared<la::cpu::tensor<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::cpu::tensor<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        m_grad({ell, std::min<int>(head_time + shift, m.size(1) - 1)}) += g;
    }

    void right_boundary_score::grad() const
    {
        autodiff::eval_vertex(score, autodiff::grad_funcs);
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

        left_end = autodiff::mul(tensor_tree::get_var(param->children[1]),
            tensor_tree::get_var(param->children[0]));
        right_end = autodiff::mul(tensor_tree::get_var(param->children[3]),
            tensor_tree::get_var(param->children[2]));

        pre_left = autodiff::mul(frames, tensor_tree::get_var(param->children[0]));
        pre_right = autodiff::mul(frames, tensor_tree::get_var(param->children[2]));
        pre_label = autodiff::mul(tensor_tree::get_var(param->children[4]),
            tensor_tree::get_var(param->children[5]));
        pre_length = autodiff::mul(tensor_tree::get_var(param->children[6]),
            tensor_tree::get_var(param->children[7]));
    }

    double segrnn_score::operator()(ifst::fst const& f,
        int e) const
    {
        autodiff::computation_graph& comp_graph = *frames->graph;

        int begin_size = comp_graph.vertices.size();

        auto& m = autodiff::get_output<la::cpu::tensor_like<double>>(frames);
        auto& length_param = autodiff::get_output<la::cpu::tensor_like<double>>(
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

        if (std::min<int>(m.size(0) - 1, head_time) == m.size(0) - 1) {
            right_embedding = autodiff::add(std::vector<std::shared_ptr<autodiff::op_t>>{ right_end });
        } else {
            right_embedding = autodiff::row_at(pre_right, std::min<int>(m.size(0) - 1, head_time));
        }

        auto label_embedding = autodiff::row_at(pre_label, ell);
        auto length_embedding = autodiff::row_at(pre_length,
            std::min<int>(int(std::log(head_time - tail_time) / std::log(1.6)) + 1,
                 length_param.size(0) - 1));

        auto& theta = autodiff::get_output<la::cpu::tensor<double>>(
            tensor_tree::get_var(param->children[11]));

        la::cpu::tensor<double> mask_vec;

        if (dropout == 0.0) {
            mask_vec.resize({theta.vec_size()}, 1);
        } else {
            mask_vec.resize({theta.vec_size()});
            std::bernoulli_distribution dist {1 - dropout};

            for (int i = 0; i < mask_vec.vec_size(); ++i) {
                mask_vec({i}) = dist(*gen) / (1.0 - dropout);
            }
        }

        auto mask = comp_graph.var(mask_vec);

        if (e >= edge_scores.size()) {
            edge_scores.resize(e + 1, nullptr);
        }

        auto edge_feat = autodiff::relu(
            autodiff::add(std::vector<std::shared_ptr<autodiff::op_t>> {
                left_embedding,
                right_embedding,
                label_embedding,
                length_embedding,
                tensor_tree::get_var(param->children[8])
            })
        );

        auto s_e = autodiff::dot(tensor_tree::get_var(param->children[11]),
            autodiff::emul(mask,
                autodiff::tanh(
                    autodiff::add(
                        tensor_tree::get_var(param->children[10]),
                        autodiff::mul(
                            edge_feat,
                            tensor_tree::get_var(param->children[9])
                        )
                    )
                )
            ));

        edge_scores[e] = s_e;

        int end_size = comp_graph.vertices.size();

        topo_shift = end_size - begin_size;

        return autodiff::get_output<double>(s_e);
    }

    void segrnn_score::accumulate_grad(double g, ifst::fst const& f,
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

        for (int i = 0; i < edge_scores.size(); ++i) {
            std::shared_ptr<autodiff::op_t> t = edge_scores.at(i);

            if (t != nullptr && t->grad != nullptr) {
                std::vector<std::shared_ptr<autodiff::op_t>> topo_order;
                for (int k = t->id; k > t->id - topo_shift; --k) {
                    topo_order.push_back(comp_graph.vertices.at(k));
                }
                autodiff::grad(topo_order, autodiff::grad_funcs);
            }
        }

        auto guarded_grad = [&](std::shared_ptr<autodiff::op_t> t) {
            if (t->grad != nullptr) {
                autodiff::eval_vertex(t, autodiff::grad_funcs);
            } else {
                std::cout << "warning: no grad." << std::endl;
            }
        };

        guarded_grad(pre_left);
        guarded_grad(pre_right);
        guarded_grad(left_end);
        guarded_grad(right_end);
        guarded_grad(pre_label);
        guarded_grad(pre_length);
    }

    length_score::length_score(std::shared_ptr<autodiff::op_t> param)
        : param(param)
    {}

    double length_score::operator()(ifst::fst const& f,
        int e) const
    {
        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        auto& m = autodiff::get_output<la::cpu::tensor<double>>(param);

        return m({ell, std::min<int>(head_time - tail_time - 1, m.size(1) - 1)});
    }

    void length_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::cpu::tensor<double>>(param);

        if (param->grad == nullptr) {
            la::cpu::tensor<double> m_grad;
            la::cpu::resize_as(m_grad, m);
            param->grad = std::make_shared<la::cpu::tensor<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::cpu::tensor<double>>(param);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        m_grad({ell, std::min<int>(head_time - tail_time - 1, m.size(1) - 1)}) += g;
    }

    bias0_score::bias0_score(std::shared_ptr<autodiff::op_t> param)
        : param(param)
    {}

    double bias0_score::operator()(ifst::fst const& f,
        int e) const
    {
        auto& v = autodiff::get_output<la::cpu::tensor<double>>(param);

        return v({0});
    }

    void bias0_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        auto& v = autodiff::get_output<la::cpu::tensor<double>>(param);

        if (param->grad == nullptr) {
            la::cpu::tensor<double> v_grad;
            v_grad.resize({v.vec_size()});
            param->grad = std::make_shared<la::cpu::tensor<double>>(std::move(v_grad));
        }

        auto& v_grad = autodiff::get_grad<la::cpu::tensor<double>>(param);

        v_grad({0}) += g;
    }

    bias1_score::bias1_score(std::shared_ptr<autodiff::op_t> param)
        : param(param)
    {}

    double bias1_score::operator()(ifst::fst const& f,
        int e) const
    {
        int ell = f.output(e) - 1;

        auto& v = autodiff::get_output<la::cpu::tensor<double>>(param);

        return v({ell});
    }

    void bias1_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        auto& v = autodiff::get_output<la::cpu::tensor<double>>(param);

        if (param->grad == nullptr) {
            la::cpu::tensor<double> v_grad;
            v_grad.resize({v.vec_size()});
            param->grad = std::make_shared<la::cpu::tensor<double>>(std::move(v_grad));
        }

        auto& v_grad = autodiff::get_grad<la::cpu::tensor<double>>(param);

        int ell = f.output(e) - 1;

        v_grad({ell}) += g;
    }

    logsoftmax_score::logsoftmax_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames)
        : param(param), frames(frames)
    {
        auto& t = autodiff::get_output<la::cpu::tensor_like<double>>(param);
        first = autodiff::reshape(param, {t.size(0), t.size(1) * t.size(2)});
        score = autodiff::mul(frames, first);
        prob = autodiff::logsoftmax(score);
    }

    double logsoftmax_score::operator()(ifst::fst const& f,
        int e) const
    {
        auto& t = autodiff::get_output<la::cpu::tensor_like<double>>(param);
        auto& m = autodiff::get_output<la::cpu::tensor_like<double>>(prob);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));
        int dur = head_time - tail_time;

        return m({head_time - 1, (int)(ell * t.size(2) + dur - 1)});
    }

    void logsoftmax_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        auto& t = autodiff::get_output<la::cpu::tensor_like<double>>(param);
        auto& m = autodiff::get_output<la::cpu::tensor_like<double>>(prob);

        if (prob->grad == nullptr) {
            la::cpu::tensor<double> m_grad;
            la::cpu::resize_as(m_grad, m);
            prob->grad = std::make_shared<la::cpu::tensor<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::cpu::tensor_like<double>>(prob);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));
        int dur = head_time - tail_time;

        m_grad({head_time - 1, (int)(ell * t.size(2) + dur - 1)}) += g;
    }

    void logsoftmax_score::grad() const
    {
        auto& g = *prob->graph;

        for (int i = prob->id; i >= first->id; --i) {
            autodiff::eval_vertex(g.vertices[i], autodiff::grad_funcs);
        }
    }

    label_logsoftmax_score::label_logsoftmax_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames)
        : param(param), frames(frames)
    {
        score = autodiff::mul(frames, param);
        prob = autodiff::logsoftmax(score);
    }

    double label_logsoftmax_score::operator()(ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::cpu::tensor_like<double>>(prob);

        int ell = f.output(e) - 1;
        int head_time = f.time(f.head(e));

        return m({head_time - 1, ell});
    }

    void label_logsoftmax_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::cpu::tensor_like<double>>(prob);

        if (prob->grad == nullptr) {
            la::cpu::tensor<double> m_grad;
            la::cpu::resize_as(m_grad, m);
            prob->grad = std::make_shared<la::cpu::tensor<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::cpu::tensor_like<double>>(prob);

        int ell = f.output(e) - 1;
        int head_time = f.time(f.head(e));

        m_grad({head_time - 1, ell}) += g;
    }

    void label_logsoftmax_score::grad() const
    {
        autodiff::eval_vertex(prob, autodiff::grad_funcs);
        autodiff::eval_vertex(score, autodiff::grad_funcs);
    }

    length_logsoftmax_score::length_logsoftmax_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames)
        : param(param), frames(frames)
    {
        score = autodiff::mul(frames, param);
        prob = autodiff::logsoftmax(score);
    }

    double length_logsoftmax_score::operator()(ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::cpu::tensor_like<double>>(prob);

        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        return m({head_time - 1, head_time - tail_time - 1});
    }

    void length_logsoftmax_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::cpu::tensor_like<double>>(prob);

        if (prob->grad == nullptr) {
            la::cpu::tensor<double> m_grad;
            la::cpu::resize_as(m_grad, m);
            prob->grad = std::make_shared<la::cpu::tensor<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::cpu::tensor_like<double>>(prob);

        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        m_grad({head_time - 1, head_time - tail_time - 1}) += g;
    }

    void length_logsoftmax_score::grad() const
    {
        autodiff::eval_vertex(prob, autodiff::grad_funcs);
        autodiff::eval_vertex(score, autodiff::grad_funcs);
    }

}
