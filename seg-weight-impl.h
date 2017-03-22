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

}
