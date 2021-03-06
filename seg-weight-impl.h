namespace seg {

    template <class fst_t, class T>
    weight_wrapper<fst_t, T>::weight_wrapper(T func)
        : func(func)
    {}

    template <class fst_t, class T>
    double weight_wrapper<fst_t, T>::operator()(fst_t const& f,
        typename fst_t::edge e) const
    {
        return func(f, e);
    }

    template <class fst_t, class T>
    std::shared_ptr<weight_wrapper<fst_t, T>> make_weight(T&& t)
    {
        return std::make_shared<weight_wrapper<fst_t, T>>(weight_wrapper<fst_t, T>(std::move(t)));
    }

    template <class fst>
    double composite_weight<fst>::operator()(fst const& f,
        typename fst::edge e) const
    {
        double sum = 0;

        for (auto& w: weights) {
            sum += (*w)(f, e);
        }

        return sum;
    }

    template <class fst>
    void composite_weight<fst>::accumulate_grad(double g, fst const& f,
        typename fst::edge e) const
    {
        for (auto& w: weights) {
            w->accumulate_grad(g, f, e);
        }
    }

    template <class fst>
    void composite_weight<fst>::grad() const
    {
        for (auto& w: weights) {
            w->grad();
        }
    }

    template <class fst>
    cached_weight<fst>::cached_weight(std::shared_ptr<seg_weight<fst>> weight)
        : weight(weight)
    {}

    template <class fst>
    double cached_weight<fst>::operator()(fst const& f,
        typename fst::edge e) const
    {
#if OMP_SAFE
        if (score_cache == nullptr) {
            auto const& edges = f.edges();

            std::vector<double> score;
            score.resize(edges.size());

            std::unordered_map<typename fst::edge, int> indices;

            for (int i = 0; i < edges.size(); ++i) {
                indices[edges[i]] = i;
            }

            #pragma omp parallel for
            for (int i = 0; i < edges.size(); ++i) {
                score[i] = (*weight)(f, edges[i]);
            }

            score_cache = std::make_shared<std::vector<double>>(score);
            indices_cache = std::make_shared<std::unordered_map<typename fst::edge, int>>(indices);
        }

        return score_cache->at(indices_cache->at(e));
#else
        double result;

        if (!ebt::in(e, score_cache)) {
            result = (*weight)(f, e);
            score_cache[e] = result;
        } else {
            result = score_cache.at(e);
        }

        return result;
#endif
    }

    template <class fst>
    void cached_weight<fst>::accumulate_grad(double g, fst const& f,
        typename fst::edge e) const
    {
        weight->accumulate_grad(g, f, e);
    }

    template <class fst>
    void cached_weight<fst>::grad() const
    {
        weight->grad();
    }

}
