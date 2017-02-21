#include "seg/seg-weight.h"

namespace seg {

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
        auto& m = autodiff::get_output<la::tensor_like<double>>(frames);

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
        auto& m = autodiff::get_output<la::tensor_like<double>>(frames);

        if (frames->grad == nullptr) {
            la::tensor<double> m_grad;
            la::resize_as(m_grad, m);
            frames->grad = std::make_shared<la::tensor<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::tensor_like<double>>(frames);

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
        autodiff::eval_vertex(score, autodiff::eval_funcs);
    }

    double frame_avg_score::operator()(ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::tensor_like<double>>(score);

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
        auto& m = autodiff::get_output<la::tensor_like<double>>(score);

        if (score->grad == nullptr) {
            la::tensor<double> m_grad;
            la::resize_as(m_grad, m);
            score->grad = std::make_shared<la::tensor<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::tensor_like<double>>(score);

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
        autodiff::eval_vertex(score, autodiff::eval_funcs);
    }

    double frame_samples_score::operator()(ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::tensor<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        int dur = head_time - tail_time;

        return m({ell, int(tail_time + dur * scale)});
    }

    void frame_samples_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::tensor<double>>(score);

        if (score->grad == nullptr) {
            la::tensor<double> m_grad;
            la::resize_as(m_grad, m);
            score->grad = std::make_shared<la::tensor<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::tensor<double>>(score);

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
        autodiff::eval_vertex(score, autodiff::eval_funcs);
    }

    double left_boundary_score::operator()(ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::tensor<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        return m({ell, std::max<int>(tail_time + shift, 0)});
    }

    void left_boundary_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::tensor<double>>(score);

        if (score->grad == nullptr) {
            la::tensor<double> m_grad;
            la::resize_as(m_grad, m);
            score->grad = std::make_shared<la::tensor<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::tensor<double>>(score);

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
        autodiff::eval_vertex(score, autodiff::eval_funcs);
    }

    double right_boundary_score::operator()(ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::tensor<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        return m({ell, std::min<int>(head_time + shift, m.size(1) - 1)});
    }

    void right_boundary_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::tensor<double>>(score);

        if (score->grad == nullptr) {
            la::tensor<double> m_grad;
            la::resize_as(m_grad, m);
            score->grad = std::make_shared<la::tensor<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::tensor<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        m_grad({ell, std::min<int>(head_time + shift, m.size(1) - 1)}) += g;
    }

    void right_boundary_score::grad() const
    {
        autodiff::eval_vertex(score, autodiff::grad_funcs);
    }

    ds_frame_avg_score::ds_frame_avg_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames)
        : param(param), frames(frames)
    {
        score = autodiff::rtmul(param, frames);
        autodiff::eval_vertex(score, autodiff::eval_funcs);
    }

    double ds_frame_avg_score::operator()(ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::tensor_like<double>>(score);

        double sum = 0;

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        for (int t = tail_time; t < head_time; ++t) {
            sum += m({ell, t});
        }

        return (head_time <= tail_time ? 0 : sum);
    }

    void ds_frame_avg_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::tensor_like<double>>(score);

        if (score->grad == nullptr) {
            la::tensor<double> m_grad;
            la::resize_as(m_grad, m);
            score->grad = std::make_shared<la::tensor<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::tensor_like<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        for (int t = tail_time; t < head_time; ++t) {
            m_grad({ell, t}) += g;
        }
    }

    void ds_frame_avg_score::grad() const
    {
        autodiff::eval_vertex(score, autodiff::grad_funcs);
    }

    ds_frame_samples_score::ds_frame_samples_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames, double scale)
        : param(param), frames(frames), scale(scale)
    {
        score = autodiff::rtmul(param, frames);
        autodiff::eval_vertex(score, autodiff::eval_funcs);
    }

    double ds_frame_samples_score::operator()(ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::tensor<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        int dur = head_time - tail_time;

        return m({ell, int(tail_time + dur * scale)}) * dur;
    }

    void ds_frame_samples_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::tensor<double>>(score);

        if (score->grad == nullptr) {
            la::tensor<double> m_grad;
            la::resize_as(m_grad, m);
            score->grad = std::make_shared<la::tensor<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::tensor<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        int dur = head_time - tail_time;

        m_grad({ell, int(tail_time + dur * scale)}) += g * dur;
    }

    void ds_frame_samples_score::grad() const
    {
        autodiff::eval_vertex(score, autodiff::grad_funcs);
    }

    ds_left_boundary_score::ds_left_boundary_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames, int shift)
        : param(param), frames(frames), shift(shift)
    {
        score = autodiff::rtmul(param, frames);
        autodiff::eval_vertex(score, autodiff::eval_funcs);
    }

    double ds_left_boundary_score::operator()(ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::tensor<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        return m({ell, std::max<int>(tail_time + shift, 0)}) * (head_time - tail_time);
    }

    void ds_left_boundary_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::tensor<double>>(score);

        if (score->grad == nullptr) {
            la::tensor<double> m_grad;
            la::resize_as(m_grad, m);
            score->grad = std::make_shared<la::tensor<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::tensor<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        m_grad({ell, std::max<int>(tail_time + shift, 0)}) += g * (head_time - tail_time);
    }

    void ds_left_boundary_score::grad() const
    {
        autodiff::eval_vertex(score, autodiff::grad_funcs);
    }

    ds_right_boundary_score::ds_right_boundary_score(std::shared_ptr<autodiff::op_t> param,
            std::shared_ptr<autodiff::op_t> frames, int shift)
        : param(param), frames(frames), shift(shift)
    {
        score = autodiff::rtmul(param, frames);
        autodiff::eval_vertex(score, autodiff::eval_funcs);
    }

    double ds_right_boundary_score::operator()(ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::tensor<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        return m({ell, std::min<int>(head_time + shift, m.size(1) - 1)}) * (head_time - tail_time);
    }

    void ds_right_boundary_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::tensor<double>>(score);

        if (score->grad == nullptr) {
            la::tensor<double> m_grad;
            la::resize_as(m_grad, m);
            score->grad = std::make_shared<la::tensor<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::tensor<double>>(score);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        m_grad({ell, std::min<int>(head_time + shift, m.size(1) - 1)}) += g * (head_time - tail_time);
    }

    void ds_right_boundary_score::grad() const
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
        if (edge_scores.size() > e && edge_scores[e] != nullptr && edge_scores[e]->output != nullptr) {
            return autodiff::get_output<double>(edge_scores[e]);
        }

        autodiff::computation_graph& comp_graph = *frames->graph;

        int begin_size = comp_graph.vertices.size();

        auto& m = autodiff::get_output<la::tensor_like<double>>(frames);
        auto& length_param = autodiff::get_output<la::tensor_like<double>>(
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

        auto& theta = autodiff::get_output<la::tensor<double>>(
            tensor_tree::get_var(param->children[11]));

        la::tensor<double> mask_vec;

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

    ds_segrnn_score::ds_segrnn_score(std::shared_ptr<tensor_tree::vertex> param,
        std::shared_ptr<autodiff::op_t> frames)
        : ds_segrnn_score(param, frames, 0.0, nullptr)
    {
    }

    ds_segrnn_score::ds_segrnn_score(std::shared_ptr<tensor_tree::vertex> param,
        std::shared_ptr<autodiff::op_t> frames,
        double dropout,
        std::default_random_engine *gen)
        : param(param), frames(frames), dropout(dropout), gen(gen)
    {
        assert(0.0 <= dropout && dropout <= 1.0);

        autodiff::computation_graph& graph = *frames->graph;

        std::shared_ptr<autodiff::op_t> frames_tmp = graph.var();

        left_end = autodiff::mul(tensor_tree::get_var(param->children[1]), tensor_tree::get_var(param->children[0]));
        right_end = autodiff::mul(tensor_tree::get_var(param->children[3]), tensor_tree::get_var(param->children[2]));

        pre_left = autodiff::mul(frames_tmp, tensor_tree::get_var(param->children[0]));
        pre_right = autodiff::mul(frames_tmp, tensor_tree::get_var(param->children[2]));
        pre_label = autodiff::mul(tensor_tree::get_var(param->children[4]),
            tensor_tree::get_var(param->children[5]));
        pre_length = autodiff::mul(tensor_tree::get_var(param->children[6]),
            tensor_tree::get_var(param->children[7]));

        auto left_embedding = autodiff::row_at(pre_left, 0);
        auto right_embedding = autodiff::row_at(pre_right, 0);
        auto label_embedding = autodiff::row_at(pre_label, 0);
        auto length_embedding = autodiff::row_at(pre_length, 0);
        auto mask = graph.var();

        auto edge_feat = autodiff::relu(
            autodiff::add(std::vector<std::shared_ptr<autodiff::op_t>> {
                left_embedding,
                right_embedding,
                label_embedding,
                length_embedding,
                tensor_tree::get_var(param->children[8])
            })
        );

        score = autodiff::dot(tensor_tree::get_var(param->children[11]),
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

        std::unordered_set<std::shared_ptr<autodiff::op_t>> exclude {
            pre_left, pre_right, pre_label, pre_length, left_end, right_end, frames_tmp };

        for (int i = 0; i <= 11; ++i) {
            exclude.insert(tensor_tree::get_var(param->children[i]));
        }

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

    double ds_segrnn_score::operator()(ifst::fst const& f,
        int e) const
    {
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        if (edge_scores.size() > e && edge_scores[e] != nullptr && edge_scores[e]->output != nullptr) {
            return autodiff::get_output<double>(edge_scores[e]) * (head_time - tail_time);
        }

        autodiff::computation_graph& comp_graph = *frames->graph;

        auto& m = autodiff::get_output<la::tensor_like<double>>(frames);
        auto& length_param = autodiff::get_output<la::tensor_like<double>>(
            tensor_tree::get_var(param->children[6]));

        int ell = f.output(e) - 1;
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

        auto& theta = autodiff::get_output<la::tensor<double>>(tensor_tree::get_var(param->children[11]));
        la::tensor<double> mask_vec;

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
            edge_feat.resize(e + 1, nullptr);
        }

        edge_feat[e] = autodiff::relu(
            autodiff::add(std::vector<std::shared_ptr<autodiff::op_t>> {
                left_embedding,
                right_embedding,
                label_embedding,
                length_embedding,
                tensor_tree::get_var(param->children[8])
            })
        );

        std::shared_ptr<autodiff::op_t> s_e = autodiff::dot(tensor_tree::get_var(param->children[11]),
            autodiff::emul(mask,
                autodiff::tanh(
                    autodiff::add(
                        tensor_tree::get_var(param->children[10]),
                        autodiff::mul(
                            edge_feat[e],
                            tensor_tree::get_var(param->children[9])
                        )
                    )
                )
            ));

        edge_scores[e] = s_e;

        std::vector<std::shared_ptr<autodiff::op_t>> topo_order;
        int d = s_e->id - score->id;
        for (auto& i: topo_order_shift) {
            topo_order.push_back(comp_graph.vertices[i + d]);
        }

        autodiff::eval(topo_order, autodiff::eval_funcs);

        return autodiff::get_output<double>(s_e) * (head_time - tail_time);
    }

    void ds_segrnn_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        std::shared_ptr<autodiff::op_t> s_e = edge_scores[e];
        if (s_e->grad == nullptr) {
            s_e->grad = std::make_shared<double>(0);
        }

        autodiff::get_grad<double>(s_e) += g * (head_time - tail_time);
    }

    void ds_segrnn_score::grad() const
    {
        autodiff::computation_graph& comp_graph = *frames->graph;

        for (int i = 0; i < edge_scores.size(); ++i) {
            std::shared_ptr<autodiff::op_t> t = edge_scores.at(i);

            if (t != nullptr && t->grad != nullptr) {
                std::vector<std::shared_ptr<autodiff::op_t>> topo_order;
                int d = t->id - score->id;
                for (auto& i: topo_order_shift) {
                    topo_order.push_back(comp_graph.vertices[i + d]);
                }
                autodiff::grad(topo_order, autodiff::grad_funcs);
            }
        }

        auto guarded_grad = [&](std::shared_ptr<autodiff::op_t> t) {
            if (t->grad != nullptr) {
                autodiff::eval_vertex(t, autodiff::grad_funcs);
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

        auto& m = autodiff::get_output<la::tensor<double>>(param);

        return m({ell, std::min<int>(head_time - tail_time - 1, m.size(1) - 1)});
    }

    void length_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::tensor<double>>(param);

        if (param->grad == nullptr) {
            la::tensor<double> m_grad;
            la::resize_as(m_grad, m);
            param->grad = std::make_shared<la::tensor<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::tensor<double>>(param);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        m_grad({ell, std::min<int>(head_time - tail_time - 1, m.size(1) - 1)}) += g;
    }

    ds_length_score::ds_length_score(std::shared_ptr<autodiff::op_t> param)
        : param(param)
    {}

    double ds_length_score::operator()(ifst::fst const& f,
        int e) const
    {
        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        auto& m = autodiff::get_output<la::tensor<double>>(param);

        return m({ell, std::min<int>(head_time - tail_time - 1, m.size(1) - 1)}) * (head_time - tail_time);
    }

    void ds_length_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        auto& m = autodiff::get_output<la::tensor<double>>(param);

        if (param->grad == nullptr) {
            la::tensor<double> m_grad;
            la::resize_as(m_grad, m);
            param->grad = std::make_shared<la::tensor<double>>(std::move(m_grad));
        }

        auto& m_grad = autodiff::get_grad<la::tensor<double>>(param);

        int ell = f.output(e) - 1;
        int tail_time = f.time(f.tail(e));
        int head_time = f.time(f.head(e));

        m_grad({ell, std::min<int>(head_time - tail_time - 1, m.size(1) - 1)}) += g * (head_time - tail_time);
    }

    bias0_score::bias0_score(std::shared_ptr<autodiff::op_t> param)
        : param(param)
    {}

    double bias0_score::operator()(ifst::fst const& f,
        int e) const
    {
        auto& v = autodiff::get_output<la::tensor<double>>(param);

        return v({0});
    }

    void bias0_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        auto& v = autodiff::get_output<la::tensor<double>>(param);

        if (param->grad == nullptr) {
            la::tensor<double> v_grad;
            v_grad.resize({v.vec_size()});
            param->grad = std::make_shared<la::tensor<double>>(std::move(v_grad));
        }

        auto& v_grad = autodiff::get_grad<la::tensor<double>>(param);

        v_grad({0}) += g;
    }

    bias1_score::bias1_score(std::shared_ptr<autodiff::op_t> param)
        : param(param)
    {}

    double bias1_score::operator()(ifst::fst const& f,
        int e) const
    {
        int ell = f.output(e) - 1;

        auto& v = autodiff::get_output<la::tensor<double>>(param);

        return v({ell});
    }

    void bias1_score::accumulate_grad(double g, ifst::fst const& f,
        int e) const
    {
        auto& v = autodiff::get_output<la::tensor<double>>(param);

        if (param->grad == nullptr) {
            la::tensor<double> v_grad;
            v_grad.resize({v.vec_size()});
            param->grad = std::make_shared<la::tensor<double>>(std::move(v_grad));
        }

        auto& v_grad = autodiff::get_grad<la::tensor<double>>(param);

        int ell = f.output(e) - 1;

        v_grad({ell}) += g;
    }

}
